# Known Issues

Documented issues that are understood but not yet fixed.  Each entry
contains enough information to implement the fix without re-investigation.

---

## Issue 1 — `test_core_math_equiv` fails on CUDA backend

**Status**: failing on every run (deterministic, not flaky)
**Affected test**: `test_core_math_equiv` (label `smoke`)
**Affected preset**: `cuda` (GPU backend); `omp` passes

### Symptom

```
FAILED: REQUIRE( to_bits(d_out_z[i]) == to_bits(h_rz[i]) )
  GPU  result: 0x447fbfff  (1022.9999389648438)
  Host result: 0x447fc000  (1023.0)
  element 1

FAILED: REQUIRE( to_bits(oy[i]) == to_bits(h_ry[i]) )
  GPU  result: 0x3d95f61b  (0.07322331517...)
  Host result: 0x3d95f61a  (0.07322330772...)
  element 1
```

Both failures are 1 ULP apart.

### Root cause

The CUDA JIT compiler (`ptxas`) contracts `a * b - c * d` into a fused
multiply-add (FMA) instruction when the result permits it.  The host
compiler (`clang++` via AdaptiveCpp) does **not** contract the same
expression in the host-side reference loop.  The two execution paths
therefore round at different points, producing a 1-ULP difference in
the result.

**Triggering computation — cross product, element 1:**

```
ax=1.1f, ay=1.2f, bx=1023.0f, by=2046.0f
rz = ax*by - ay*bx             // = 1.1*2046 - 1.2*1023
```

- Host (two-op, with intermediate rounding):
  `f32(1.1f * 2046.0f) - f32(1.2f * 1023.0f) = 2250.6... - 1227.6... = 1023.0`
- GPU (FMA, single rounding):
  `fma(1.1f, 2046.0f, -f32(1.2f * 1023.0f)) ≈ 1022.999...`

The exact result is 1023.000000... so the host's two-op path rounds to
1023.0 while the FMA path rounds to the next-lower float.

`Vec3::cross` implementation (`core/include/core/math/vec3.hpp:27`):
```cpp
return {y * o.z - z * o.y,
        z * o.x - x * o.z,
        x * o.y - y * o.x};   // ← contracted to FMA on GPU
```

`Quat::rotate` has the same pattern via two nested cross products
(`core/include/core/math/quat.hpp:39–41`).

### Fix options (choose one)

**Option A — Relax the test to 1-ULP tolerance (recommended)**

The test's goal is to confirm that the device math types produce results
consistent with host scalar math, not to assert bit-for-bit identical
floating-point schedules.  A 1-ULP tolerance matches the IEEE 754
guarantee for individually-rounded operations.

In `core/tests/math_equiv_test.cpp`, replace the `to_bits` bit-cast
checks with `WithinULP`:

```cpp
// before
REQUIRE(to_bits(d_out_z[i]) == to_bits(h_rz[i]));

// after
REQUIRE_THAT(d_out_z[i], Catch::Matchers::WithinULP(h_rz[i], 1));
```

Apply to all three components of both `Vec3::cross` and `Quat::rotate`
equivalence tests (lines 67–72 and 116–121).

**Option B — Disable FMA contraction in the kernel**

Add `-ffp-contract=off` to the SYCL device compilation flags so the
GPU path matches the host path.  Locate the flag in
`cmake/BackendOptions.cmake` or `CMakeLists.txt` under
`add_sycl_to_target`.  This sacrifices the free precision benefit of
FMA everywhere, not just in the test.

**Option C — Force FMA on the host reference too**

Rewrite the host reference loop using `std::fma` so both paths use FMA:

```cpp
h_rz[i] = std::fma(h_ax[i], h_by[i], -h_ay[i] * h_bx[i]);
```

Exact only when `ay*bx` is computed first; still not bit-identical with
GPU in general.  More fragile than Option A.

**Recommendation**: Option A.  The test was written before the FMA
behaviour was understood; `WithinULP(1)` is the correct tolerance for
operations that are "correct to last-place" on either path.

---

## Issue 2 — 1025-body stress test runs at 51 fps (0.86× realtime)

**Status**: measured on RTX 3060, Release build, `cuda` preset
**Target**: v0.1 demo goal is ≥ realtime (≥60 fps) for 1025 bodies
**Current**: `stress_test_blocks` with 1025 bodies, 600 frames → **51 fps**

Three independent sub-causes, each fixable in isolation.

---

### Issue 2a — Broadphase pair sort dominates GPU time (77%)

**Stage profiling on 512 bodies** (from `test_core_bench`, CUDA):

| Stage                   | Mean latency | Share |
|-------------------------|--------------|-------|
| broadphase sort_pairs   | 4.26 ms      | 77%   |
| broadphase build+query  | 0.90 ms      | 16%   |
| integrator              | 0.18 ms       | 3%    |
| narrowphase             | 0.12 ms      | 2%    |
| xpbd solver             | 0.05 ms      | 1%    |

**Root cause**: the sort is implemented as a host-orchestrated bitonic
network in `compute/include/compute/sort.hpp`.  Each pass of the
network is a separate `q.parallel_for(...)` submission.  For n pairs,
there are `log2(n) * (log2(n)+1) / 2` kernel launches:

| Actual pair count | n\_sort (next pow2) | Kernel launches |
|-------------------|---------------------|-----------------|
| ~500              | 512                 | 45              |
| ~1 024            | 1 024               | 55              |
| ~3 100            | 4 096               | 78              |
| 65 536            | 65 536              | 136             |

Each kernel launch incurs PCIe round-trip + CUDA launch overhead.
At ~77 µs per launch (observed), 78 launches = ~6 ms for the sort
alone — independent of how little work each kernel actually does.

**Relevant code**:
- Sort implementation: `compute/include/compute/sort.hpp:15–38`
- Call site: `core/src/broadphase.cpp:352–363` (`sort_pairs`)

**Fix**: replace the host-loop bitonic sort with a single-dispatch
radix sort.

*Backend-portable path (preferred)*: use the SYCL 2020 Group Algorithms
or `oneapi::dpl::sort` (Intel oneDPL, available as a vcpkg package
`onedpl`).  For the `omp` backend this falls back to `std::sort`; for
`cuda`/`hip` backends it uses the respective vendor's radix sort.

```cmake
# vcpkg.json
{ "name": "onedpl" }
```

```cpp
#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

void sort_by_key(Stream& s, Key* keys, Value* values, size_t n) {
    auto policy = oneapi::dpl::execution::make_device_policy(s.queue());
    // zip keys+values, sort by key
    auto zipped = oneapi::dpl::make_zip_iterator(keys, values);
    oneapi::dpl::sort(policy, zipped, zipped + n,
        [](auto a, auto b){ return std::get<0>(a) < std::get<0>(b); });
}
```

*CUDA-only fallback*: `cub::DeviceRadixSort::SortPairs` (CUB is
bundled with CUDA Toolkit ≥ 11; one kernel call, ~100 µs for 64k pairs).
Only viable if the compute shim gains a backend-dispatch mechanism.

Expected speedup: from 78+ launches → 1 launch, ~4 ms → ~0.1 ms for
the sort step (40× improvement on the sort; ~3× improvement on total
frame time).

---

### Issue 2b — BVH refit kernel is sequential

**Root cause**: the LBVH refit (bottom-up AABB propagation) is
submitted as a single work-item loop:

```cpp
// core/src/broadphase.cpp:217
parallel_for(s, 1u, [=](size_t) {
    for (size_t leaf_k = 0; leaf_k < n_int; ++leaf_k) {
        // compute leaf AABB, walk up tree with atomic flags
    }
});
```

This runs the entire refit in a single GPU thread.  For 1025 bodies,
that is 1025 sequential iterations each doing a matrix multiply to
compute the box AABB plus a tree walk.  The `parallel_for(1u, ...)`
idiom exists because the original Karras refit algorithm uses
`atomic_add_seq` for sibling synchronisation — two siblings race to
write an internal node; the second one to arrive wins.  This pattern
is correct when each leaf is its own thread, but the current code
foregoes the parallelism by putting all leaves in one thread.

**Fix**: parallelise the leaf AABB computation and refit, keeping the
`atomic_add_seq` logic intact:

```cpp
// Step 5a: parallel leaf AABB computation
parallel_for(s, static_cast<size_t>(n), [=](size_t leaf_k) {
    // compute AABB for leaf_k, store in d_mn_*/d_mx_*[n-1+leaf_k]
});

// Step 5b: parallel bottom-up refit (each leaf thread walks its path)
parallel_for(s, static_cast<size_t>(n), [=](size_t leaf_k) {
    int cur = n_int - 1 + static_cast<int>(leaf_k);
    while (true) {
        int p = d_par[cur];
        if (p < 0) break;
        uint32_t old = atomic_add_seq(d_flags + p, 1u);
        if (old == 0u) break;
        // merge children, write parent, continue
        cur = p;
    }
});
```

The `atomic_add_seq` fence ensures the second-arriving sibling sees
the first sibling's AABB write before merging.  This is exactly the
Karras 2012 parallel refit design and is safe to parallelise.

**Caveat**: on the CPU (OMP) backend, `seq_cst` atomics are expensive.
For the `omp` build, keeping the sequential loop may be faster.  A
`#ifdef` on backend type or a runtime-dispatch check would allow both.

Expected speedup on GPU: linear in n for the leaf-AABB pass (1025 threads
instead of 1); tree-walk depth is O(log n) so total work is O(n log n)
but now fully parallel.

---

### Issue 2c — Extra blocking GPU sync per frame in `stress_test_blocks`

**Root cause**: `stress_test_blocks/main.cpp` calls `np.download_count(s)`
in the hot loop to accumulate `total_contacts` for the metrics JSON:

```cpp
// examples/stress_test_blocks/main.cpp:191
total_contacts += np.download_count(s);   // ← blocking memcpy+wait every frame
```

`Narrowphase::download_count` is:
```cpp
uint32_t cnt; s.queue().memcpy(&cnt, d_count_.data(), sizeof(cnt)).wait(); return cnt;
```

This is a second blocking GPU sync per frame (on top of the required
`bp.download_count(s)` on line 180).  Together with the every-other-frame
`download_state()` (7 × `memcpy.wait()`), the hot loop has far more
sync points than needed for physics correctness.

**Fix**: accumulate the contact count asynchronously.  One option is to
move the contact count download outside the loop entirely and derive an
approximate average from a single end-of-run sample:

```cpp
// remove from loop:
// total_contacts += np.download_count(s);

// after loop:
s.wait();
uint32_t last_contacts = np.download_count(s);
double avg_cnt = static_cast<double>(last_contacts);  // approximate
```

Alternatively, keep one download every N frames (e.g., every 10) and
amortise the cost.  The metrics value is informational — it does not
affect simulation correctness.

For the trajectory `download_state()` path (every-other-frame): this
is inherent to the output format and cannot be eliminated while
maintaining the current trajectory file.  Increasing the snapshot
interval (e.g., every 4 frames instead of 2) halves that overhead.

Expected speedup: removing the extra `download_count` sync removes
one full CPU-GPU round-trip per frame.  On PCIe Gen 4 with RTX 3060,
each round-trip is ~0.3–1 ms.  At 51 fps (19.6 ms/frame), removing
1 ms → ~55 fps (+8%).

---

---

## Issue 3 — `test_core_articulation` SIGSEGV on "Fixed joint" and "Revolute joint" cases

**Status**: deterministic crash on every run (OMP and CUDA)
**Affected tests**: "Fixed joint: static parent, dynamic child pulled to constraint" and
"Revolute joint: anchor constraint satisfied, axis alignment preserved"
**Not affected**: "Ball joint: violation corrected to constraint satisfaction" (runs but numerical assertion fails)

### Symptom

```
FAILED:
due to a fatal error condition:
  SIGSEGV - Segmentation violation signal
```

The segfault occurs inside the articulation solver kernel, before any assertions run.
The "Ball joint" case does not segfault but fails a `WithinAbs(0.f, 1e-3f)` check with
error ≈ 0.06 (constraint not converged).

### Root cause (to be investigated)

Likely candidates:
- Out-of-bounds joint index or body index in the articulation kernel when the body count
  changes between scenes (each test case builds a different JointStore).
- Uninitialized device memory for the parent body in the "Fixed joint" case (the
  static parent may not be uploaded to the device correctly).
- Buffer size mismatch: `JointStore` or `ArticulationSolver` constructed with an
  `n_joints` that doesn't match the actual joint count.

**Starting point**: `core/tests/articulation_test.cpp:178` (Fixed joint setup) and
`core/tests/articulation_test.cpp:221` (Revolute joint setup); the crash site is the
first kernel launch after `solver.solve(s, ...)`.

### Fix options

Investigate with a debug build + ASAN or by adding bounds assertions in the articulation
kernel before the first array access.  Likely a one-off setup error in the test or in
`ArticulationSolver::solve()`.

---

## Summary table

| ID  | Issue                                              | File(s)                                          | Effort | Impact      |
|-----|----------------------------------------------------|--------------------------------------------------|--------|-------------|
| 1   | `test_core_math_equiv` 1-ULP CUDA failure (FMA)    | `core/tests/math_equiv_test.cpp:67–72, 116–121` | XS     | Test passes |
| 2a  | Bitonic sort: O(log²n) kernel launches (77% time)  | `compute/include/compute/sort.hpp`               | M      | ~3× fps     |
| 2b  | Sequential BVH refit (`parallel_for(1u, ...)`)     | `core/src/broadphase.cpp:217`                    | S      | ~1.5× fps   |
| 2c  | Extra `np.download_count` sync per frame           | `examples/stress_test_blocks/main.cpp:191`       | XS     | ~8% fps     |
| 3   | `test_core_articulation` SIGSEGV (Fixed/Revolute)  | `core/tests/articulation_test.cpp:178, 221`      | S      | 2 tests crash |

Fix order recommendation: 1 → 2c → 3 → 2b → 2a (ascending effort, each is
independent of the others).
