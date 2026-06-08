# Known Issues

Documented issues that are understood but not yet fixed.  Each entry
contains enough information to implement the fix without re-investigation.

---

## Issue 1 — `test_core_math_equiv` fails on CUDA backend

**Status**: ✅ **RESOLVED** (2026-05-28) — bit-exact checks replaced with `WithinULP(1)` at `core/tests/math_equiv_test.cpp:67–72, 116–121`. OMP passes; CUDA 1-ULP FMA behaviour is now within tolerance.

_(Original analysis preserved below for reference.)_

**Original status**: failing on every run (deterministic, not flaky)
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

## Issue 2 — 1025-body stress test runs sub-realtime on CUDA

**Status**: ✅ **RESOLVED** (2026-06-03). Both single-work-item bottlenecks are now
parallel: narrowphase (scratch-append + key-sort + gather) **and** the XPBD solver
(graph-colored parallel Gauss-Seidel). `stress_test_blocks` on RTX 3060 now runs
**~230–237 fps warm** (Release, `cuda` preset) — up from ~55 → ~101 (narrowphase) →
~237 (solver). The GPU now beats the OpenMP CPU (~205 fps). 2a still latent for
n > 1024 pairs (minor).
**Target**: v0.1 demo goal ≥ 60 fps for 1025 bodies — **exceeded** (~237 fps warm).

### Update (2026-06-03) — XPBD solver parallelised (graph coloring)

`core/src/xpbd_solver.cpp`: the single-work-item Gauss-Seidel pass is replaced by a
**graph-colored** parallel solve. Constraints (contacts + joints) are partitioned so
no two sharing a *dynamic* body get the same colour (static/kinematic bodies,
`inv_mass==0`, are not coloring nodes — their writes are guarded no-ops, so all
box↔ground contacts collapse into a handful of colours). Within a colour, writes are
disjoint ⇒ one parallel kernel, **no atomics**; colours run in queue order ⇒
Gauss-Seidel across colours. Deterministic by construction (fixed colour order,
disjoint writes), matching NVIDIA Newton/Warp and PhysX 5 TGS. Coloring is a
single-work-item greedy pass (lowest free colour via a per-body 64-bit used-colour
mask) run once per frame and amortised over `n_iters` parallel sweeps; scenes ≤
`kSerialThreshold` (256) constraints keep the serial path (small scenes / goldens
unchanged). The solver dropped from ~9 ms to a small fraction of the frame; stress
went ~101 → ~237 fps warm. Result hashes change (colour order ≠ contact-index order);
goldens regenerated. See REPORT.md §4.5/§4.6.

### Update (2026-06-03) — narrowphase parallelised

`core/src/narrowphase.cpp`: the contact tests now run one work-item per candidate
pair, appending to a scratch store with a `(pair_idx, sub_idx)` key, then sorting
by key and gathering into the main store — reproducing the sequential pair order
so the contact set/order and determinism are preserved (mesh pairs with unbounded
contacts are handled because ordering is by key, not fixed slots). The narrowphase
stage dropped from ~9.4 ms to ~1 ms/frame; stress went ~55 → ~101 fps warm.
CPU/OpenMP output is bit-identical; CUDA contact values shift ≤1 ULP from FMA
contraction differing between the wide `parallel_for` and the old single-thread
loop (same effect as Issue 1), which deterministically changes the
manipulator_pick CUDA hash (grasp still succeeds).

The solver remains; see "Fix direction → Solver" below.

---

### Re-profile (2026-06-03) — the bottleneck is now the sequential narrowphase + solver

The Phase 5.5 broadphase work (single-kernel sort for n ≤ 1024, parallel refit)
**solved the original bottleneck**: the sort that was 77% of frame time is now 2.7%.
Re-profiling the *current* pipeline (per-stage `s.wait()` barriers, 1025 bodies,
~3 765 contacts/frame, avg over 170 warm frames) shows the cost has entirely
relocated:

| Stage           | ms/frame | Share |
|-----------------|----------|-------|
| **narrowphase** | **9.42** | **47.4%** |
| **solve (XPBD)**| **9.03** | **45.5%** |
| build_and_query | 0.80     | 4.0%  |
| sort_pairs      | 0.53     | 2.7%  |
| integrate       | 0.04     | 0.2%  |
| download_count  | 0.03     | 0.2%  |
| **total**       | **19.85**| 50.4 fps |

**Root cause**: the **narrowphase and the XPBD solver are dispatched as
single-work-item kernels** — `parallel_for(s, 1, …)` at `core/src/narrowphase.cpp:1055,
1077` and `core/src/xpbd_solver.cpp:390`. All ~3 765 contacts are processed
sequentially on **one** GPU thread (the solver does it ×10 iterations). A single
GPU core (~1.3 GHz, scalar, no ILP) is far slower than a single CPU core (~3.2 GHz,
wide out-of-order), which is exactly why OpenMP (205 fps) beats CUDA here — and why
the GPU loses at every current scene size. This is the deliberate determinism
mechanism from `REPORT.md` §4.6 (single-work-item narrowphase + solver give a fixed,
reproducible contact-processing order), now exposed as the headline cost once the
broadphase was parallelised.

**Fix direction**:
- *Narrowphase* — ✅ **done** (2026-06-03, see Update above). Parallelised via
  scratch-append + key sort + gather rather than fixed pair-indexed slots, which
  also handles mesh pairs (unbounded contacts/pair). ~9.4 ms → ~1 ms.
- *Solver* — **remaining, harder**: XPBD is Gauss–Seidel with a sequential
  dependency between contacts that share a body. Determinism + parallelism needs
  graph colouring or contact-island partitioning (parallel within a colour/island,
  fixed colour order), or a Jacobi-style sweep with under-relaxation. This is a
  solver redesign, not a dispatch change, and interacts with the determinism
  constraint (Issue 4). It is now the single largest GPU cost.

> Note: parallelising the solver interacts with determinism (Issue 4) — any parallel
> contact solve must fix a reproducible processing order, the same constraint that
> the parallel BVH refit violated.

---

### Original analysis (2026-05-28) — three sub-causes, now largely historical

---

### Issue 2a — Broadphase pair sort dominates GPU time (77%) — partially resolved

**Resolution (2026-05-28)**: For n ≤ 1024 (covers all practical pair counts for ≤ ~1000 bodies), `sort_by_key` now uses a single work-group kernel with local memory — all O(log²n) bitonic passes execute in one kernel launch. For n > 1024 it falls back to the original multi-launch bitonic. `compute/include/compute/sort.hpp` updated.

_(Original analysis preserved below.)_

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

### Issue 2b — BVH refit kernel is sequential — RESOLVED

**Resolution (2026-05-28)**: The `parallel_for(s, 1u, ...)` sequential loop in `core/src/broadphase.cpp` (Step 6) was replaced with `parallel_for(s, n_int, ...)`. Each leaf gets its own work-item; Karras atomic-flag synchronisation (`atomic_add_seq`) remains intact. The `d_flags_` buffer is already zero-initialised at Step 3 before the refit runs.

_(Original analysis preserved below.)_

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

### Issue 2c — Extra blocking GPU sync per frame in `stress_test_blocks` — RESOLVED

**Resolution (2026-05-28)**: `np.download_count(s)` removed from the hot loop in `examples/stress_test_blocks/main.cpp`. One `download_count` call remains after the loop to approximate the average contact count using the last frame's value.

_(Original analysis preserved below.)_

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

**Status**: ✅ **RESOLVED** (fixed as a side effect of Phase 5 solver and buffer-upload work)
**Resolved in**: Phase 5 (2026-05-27)

The crash was caused by uninitialized device memory: static bodies were reaching the
solver with non-zero inverse-inertia values because `body_store.cpp` did not yet zero
them for `BodyFlag::Static` bodies, and the upload sequencing (`s.wait()` placement)
was not yet in its correct form.  Both issues were corrected during Phase 5:

- `body_store.cpp`: static bodies now unconditionally get `iI_xx = … = iI_yz = 0`.
- Test helpers call `s.wait()` in the caller scope after the upload, not inside the
  helper, ensuring the kernel sees fully-initialized data.

All 6 articulation test cases pass (30 assertions) across 200 random Catch2 seeds on
the OMP Debug build.

---

## Issue 4 — CUDA non-determinism at scale (parallel BVH refit AABB-merge race)

**Status**: ✅ **RESOLVED** (2026-06-03) via fix option 3 — level-synchronised refit.
`stress_test_blocks` on CUDA now returns an **identical hash across runs**
(`3b31b708569886eb` ×3, bit-identical to the single-work-item reference). Gated by a
new 256-box determinism case in `test_core_sim_determinism`.
**Was**: open backlog. Measured on RTX 3060, Release, `cuda` preset, 2026-06-03.
**Affected**: `stress_test_blocks` (1 025 bodies) and any CUDA scene with a deep BVH.
**Not affected**: OpenMP (all sizes); CUDA small/medium scenes (`manipulator_pick`,
`arm_push`, both `test_core_sim_determinism` golden-gate scenes).

### Resolution (2026-06-03) — level-synchronised refit (fix option 3)

`core/src/broadphase.cpp` Step 6 replaces the racy atomic-rendezvous walk-up with:
(a) a parallel kernel computing every leaf AABB, then (b) repeated merge **rounds**,
each a *separate* queue-ordered `parallel_for` over internal nodes — a node merges
once both children are finalised (`d_flags_` doubles as the per-node "ready" flag;
leaves are implicitly ready). Because a parent always reads child AABBs written by a
*previous* kernel, there is no intra-kernel cross-thread read of in-flight data, so
the result is bit-identical every run while staying parallel. Rounds run in batches
of 16 and stop once the root is finalised (the Karras tree over distinct 64-bit keys
has height ≤ 64). This keeps Issue 2b's parallelism (perf back to ~237 fps, vs ~177
for the single-work-item fallback) without the determinism cost.

_(Original analysis preserved below for reference.)_

### Symptom

`stress_test_blocks` on CUDA produces a **different final-state hash on every run**,
and the demo's own `avg_contacts_per_frame` swings between runs:

```
run1 avg_contacts: 3387    hash: 516d56cec7d78bfe
run2 avg_contacts: 3384    hash: 95b6edff697b0926
run3 avg_contacts: 1707    hash: 5b14924f25a02186
```

The contact count diverges **before the solver runs**, so the non-determinism
originates in the broadphase, not the XPBD solve.

### Root cause

Issue 2b's resolution replaced the single-work-item BVH refit with a Karras
*parallel* refit (`parallel_for(s, n_int, …)` in `core/src/broadphase.cpp`, Step 6).
Each leaf walks up to the root; at each internal node an `atomic_add_seq` flag lets
the **second** arriving thread merge the two children's AABBs:

```cpp
uint32_t old = atomic_add_seq(d_flags + p, 1u);
if (old == 0u) break;                       // first arrival: bail
int lc = d_lft[p], rc = d_rgt[p];
float pmnx = min(d_mn_x[lc], d_mn_x[rc]);   // <-- reads child AABBs
...
```

The `atomic_add_seq` orders the *flag*, but the child AABB stores are **plain
(non-atomic) `sycl::half` writes**. On CUDA the device-scope atomic does not reliably
establish happens-before for those non-atomic writes (the same memory-model gap the
original single-work-item refit was written to avoid — see §4.6 of `REPORT.md`). The
second thread can therefore read a child AABB that the first thread has not yet
flushed, getting a stale/default value. That corrupts the internal node's AABB,
changing which leaf pairs the traversal reports — non-deterministically, and
differently each run. Shallow trees (small scenes) almost never hit the window;
the deep tree at 1 025 bodies does.

This is **not caught by CI**: `test_core_sim_determinism` uses an 8-box and a 2-link
scene whose BVHs are 3–4 levels deep.

### Relevant code

- Parallel refit: `core/src/broadphase.cpp` Step 6 (the `parallel_for(s, n_int, …)`
  bottom-up walk with `atomic_add_seq(d_flags + p, …)`).
- `atomic_add_seq`: `compute/include/compute/atomic.hpp`.

### Fix options

1. **Revert to single-work-item refit** (`parallel_for(s, 1u, …)` looping all leaves) —
   simplest, restores determinism, reintroduces Issue 2b's serial cost (was a small
   share of frame time vs the sort). Lowest risk.
2. **Correctly fence the parallel refit** — promote the child-AABB stores to atomics,
   or add an explicit cross-work-item fence with the right scope, so the AABB writes
   are ordered against the flag. Keeps the parallelism; needs careful validation of
   AdaptiveCpp's fence support per backend.
3. **Two-pass refit** — process the tree one level at a time with a queue-ordered
   `parallel_for` per level (separate kernels ⇒ implicit global ordering), avoiding
   intra-kernel cross-thread reads. Deterministic and parallel, at the cost of
   `tree_depth` kernel launches.

### Verification

Add a large-scene case to `test_core_sim_determinism` (e.g. 256+ bodies, run twice,
`REQUIRE(h1 == h2)`) so this class of bug is gated, then confirm
`stress_test_blocks` returns an identical hash across 3 runs on CUDA.

---

## Issue 5 — Morton-sort GPU fast path selected by compile-time macro, not device

**Status**: ✅ **RESOLVED** (2026-06-08). `compute/include/compute/sort.hpp` now gates
the single-work-group local-memory bitonic fast path on `q.get_device().is_gpu()`
at **runtime** instead of the compile-time `DYPHUR_BACKEND_IS_CUDA` macro.

### Symptom / root cause

The fast path keeps all bitonic passes on-chip with work-group barriers — correct on
GPUs, but AdaptiveCpp's OpenMP backend mis-synchronises those barriers and silently
returns an unsorted result. It was gated by a *compile-time* backend macro, so a
**CUDA build still selected the fast path when running on `Device::default_cpu()`** —
which is exactly what `test_core_sim_determinism` does. The CPU determinism gate was
therefore running a broken sort, producing an out-of-order Morton array and a
**malformed Karras tree** (observed: a 2-cycle in parent pointers, a node with
`left == right`). The old top-down stack traversal happened to tolerate it (the
detached cycle is unreachable from the root) and produced a deterministic-but-wrong
result, so the bug stayed hidden until a stackless rope traversal (which walks parent
pointers) hung on the cycle. Fix: choose the path by actual device type. The GPU still
uses the fast path; any CPU device uses the correct multi-launch path.

---

## Issue 6 — `stress_test_blocks` non-deterministic at very high body counts (≥ ~8k)

**Status**: open (backlog). Observed 2026-06-08 on RTX 3060, Release, `cuda` preset.
**Affected**: `--grid 16 32 16` (8 192 bodies) — 3 runs gave 3 different hashes.
**Not affected**: 1 024 and 4 096 bodies (deterministic run-to-run); the CI gate scenes.

### Likely cause

Tall stacks (32 boxes high) interpenetrate violently in the first frames, spiking the
per-frame candidate-pair and contact counts far above steady state. If a transient
spike exceeds a capacity bound (broadphase `max_pairs`, narrowphase contact store, or
the solver's lambda buffer), the overflowing entries are dropped via `if (idx < cap)`
guards — and *which* are dropped depends on atomic-append order, which is not
deterministic. The pair set then differs run to run. Note the determinism is lost in
the broadphase/narrowphase (count-dependent), not the solver.

### Fix options

1. Size capacities for the transient worst case (cheap; `stress_test_blocks` already
   scales them with body count — raise the multipliers), or
2. Make overflow deterministic (e.g. reserve slots by a stable key rather than a race),
   or detect overflow and grow + re-run the frame.

### Verification

Extend the determinism gate with an 8k+ stacked case once a fix lands; confirm
`stress_test_blocks --grid 16 32 16` returns an identical hash across 3 runs.

---

## Summary table

| ID  | Issue                                              | File(s)                                          | Effort | Impact      |
|-----|----------------------------------------------------|--------------------------------------------------|--------|-------------|
| 1   | ~~`test_core_math_equiv` 1-ULP CUDA failure (FMA)~~    | resolved 2026-05-28                          | —      | ✅ resolved |
| 2   | ~~CUDA stress sub-realtime~~ — narrowphase **and** XPBD solver now parallel (~55→~101→~237 fps; GPU beats CPU) | ~~`narrowphase.cpp`~~, ~~`xpbd_solver.cpp`~~ done | —      | ✅ resolved |
| 2a  | Bitonic sort: O(log²n) kernel launches (n > 1024)  | `compute/include/compute/sort.hpp`               | M      | now ~6–8% of frame at 8k |
| 2b  | ~~Sequential BVH refit~~                           | resolved 2026-05-28 (caused Issue 4)             | —      | ✅ resolved |
| 2c  | ~~Extra `np.download_count` sync per frame~~        | resolved 2026-05-28                              | —      | ✅ resolved |
| 3   | ~~`test_core_articulation` SIGSEGV (Fixed/Revolute)~~  | resolved in Phase 5                          | —      | ✅ resolved  |
| 4   | ~~CUDA non-determinism at scale (parallel BVH refit AABB race)~~ | resolved 2026-06-03 (level-sync refit) | —      | ✅ resolved |
| 5   | ~~Morton-sort fast path gated by compile-time macro (broke CPU determinism path)~~ | resolved 2026-06-08 (runtime device gate) | — | ✅ resolved |
| 6   | `stress_test_blocks` non-deterministic at ≥ ~8k bodies (transient capacity overflow) | `examples/stress_test_blocks/main.cpp`, broadphase/narrowphase caps | M | correctness (extreme scenes only) |

Remaining open:
- **Issue 2a** — n > 1024 pair sort still multi-launch (oneDPL/CUB or radix sort);
  minor now (the colored solver, not the sort, is the scaling story).
- **Issue 6** — transient capacity overflow makes ≥ ~8k-body stacks non-deterministic.
