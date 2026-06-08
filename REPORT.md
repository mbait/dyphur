# dyphur — Technical Report

**Project**: GPU-first rigid-body physics framework for robotics simulation  
**Developer**: Alexander Solovets (asolovets@gmail.com) — solo, full-time  
**License**: Apache-2.0  
**Reference hardware**: NVIDIA GeForce RTX 3060 (Ampere, sm_86, 12 GB VRAM) + Intel Xeon E5-2667 v4 @ 3.20 GHz  
**Status**: Phases 0–5.5 complete. Phase 6 (differentiability) deferred.

---

## 1. Motivation

Existing physics engines for robotics fall into one of three categories: GPU-accelerated but vendor-locked (NVIDIA Newton/Warp, IsaacSim), CPU-bound or weakly GPU-accelerated (Bullet, ODE, MuJoCo default CPU paths), or high-accuracy but not designed for the realtime-on-GPU regime that high-fidelity indoor robot simulation now demands. dyphur closes that gap by building GPU-first, vendor-neutral rigid-body physics using SYCL 2020 / AdaptiveCpp, with CUDA, HIP (AMD ROCm), and CPU (OpenMP) as co-equal backends.

The design target: simulate indoor scenes with 1 000+ rigid bodies involving manipulators, mobile robots, and graspable objects at realtime rates (≥ 60 Hz), deterministically, in a fully headless environment.

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Public API:  C++ headers  +  Python bindings (nanobind)             │
├──────────────────────────────────────────────────────────────────────┤
│  Scene / asset layer (scene/)                                        │
│    SDF/URDF loader (libsdformat) · V-HACD decomposition · mesh BVH  │
├──────────────────────────────────────────────────────────────────────┤
│  Physics core (core/)                                                │
│    Bodies · Articulations · Broadphase (LBVH) · Narrowphase ·       │
│    Contact manifold · XPBD solver · Integrator · Ray queries ·       │
│    Contact sensors                                                    │
├──────────────────────────────────────────────────────────────────────┤
│  Compute abstraction (compute/)                                       │
│    Device · Buffer<T> · parallel_for · reduce · sort_by_key ·        │
│    atomic_add · Stream · Event                                        │
│    Backends: CUDA · HIP · Level Zero (opt.) · OpenMP (CPU fallback)  │
├──────────────────────────────────────────────────────────────────────┤
│  Math                                                                 │
│    Host: Eigen · Device: hand-rolled SoA types (Vec3, Quat, Mat3…)  │
│    Logging: spdlog · Formatting: fmt · Profiling: Tracy              │
└──────────────────────────────────────────────────────────────────────┘
```

### Module map

| Path | Purpose |
|---|---|
| `compute/` | Thin SYCL shim: `Device`, `Buffer<T>`, `parallel_for`, bitonic sort, deterministic reduction, `atomic_add_seq`, `Stream`, `Event` |
| `core/` | Physics: integrator, LBVH broadphase, narrowphase (SAT + GJK/EPA), XPBD solver, joints, ray queries, contact sensors |
| `scene/` | SDF/URDF loader, V-HACD decomposition pipeline, mesh-BVH construction |
| `bindings/python/` | nanobind module: `Device`, `Stream`, `BodyStore`, `ShapeStore`, `Broadphase`, `Narrowphase`, `ContactSensor` |
| `examples/` | Headless demos emitting `.trajectory` + `.scene` + `.metrics.json` + `.golden` |
| `tools/viz/` | Standalone trajectory visualiser (Magnum-based, not on CI path) |
| `cmake/` | Backend selection, device-header allow-list lint, determinism harness |

---

## 3. Compute Abstraction Layer

The entire compute layer is exposed through a thin C++ header API in `compute/include/compute/`. The rest of the codebase never calls SYCL directly — only through these primitives.

**Why SYCL / AdaptiveCpp**: SYCL 2020 is the only production-ready, royalty-free standard that targets CUDA, HIP, and CPU in a single source base. AdaptiveCpp (formerly hipSYCL) implements SYCL 2020 via LLVM, with SSCP (single-source, cross-platform) compilation as the primary mode. Vendor-specific CUDA/HIP optimisations are available without forking code paths.

**Key primitives**:

- `Buffer<T>` — USM (unified shared memory) device allocation with typed `upload()` / `download()` and raw `data()` pointer for kernel capture.
- `parallel_for(Stream&, size_t n, lambda)` — submits a work-item kernel over `[0, n)`. The lambda captures `Buffer::data()` pointers (plain pointers, no smart-pointer overhead in kernels).
- `sort_by_key(Stream&, Key*, Value*, size_t n)` — bitonic sort. On **GPU backends** with `n ≤ 1024`: a single kernel launch using SYCL work-group local memory — all `O(log² n)` bitonic passes execute on-chip without PCIe round-trips. Otherwise (`n > 1024`, or any CPU/OpenMP build): a multi-launch path, one global kernel per bitonic step, ordered by the in-order queue. (The local-memory fast path is gated to GPUs because AdaptiveCpp's OpenMP backend does not reliably synchronise the work-group barriers it relies on.)
- `reduce(Stream&, T*, size_t n, T init, BinaryOp)` — parallel tree reduction.
- `atomic_add_seq(T*, T)` — sequentially-consistent fetch-add, used for lock-free counters inside kernels (e.g., broadphase pair accumulator, Karras BVH refit flags).
- `Stream` — owns an `sycl::queue`. In-order by default, giving automatic kernel serialisation within a frame without explicit events.

**Dependency policy**: STL containers, exceptions, virtual functions, dynamic allocation, and file I/O are forbidden in device code. The `cmake/DependencyAllowlist.cmake` lint target enforces this boundary at build time. Eigen is explicitly excluded from device code (its SYCL support is unofficial and intermittent).

---

## 4. Physics Pipeline

One fixed-function frame step: **integrate → broadphase → sort pairs → narrowphase → solve**.

### 4.1 Data Layout

All body state is SoA (Structure-of-Arrays) throughout hot paths. `BodyView` holds:

```cpp
float* pos_x, *pos_y, *pos_z;          // world-space position
float* rot_w, *rot_x, *rot_y, *rot_z;  // unit quaternion
float* vel_x, *vel_y, *vel_z;          // linear velocity
float* ang_x, *ang_y, *ang_z;          // angular velocity
float* inv_mass;
float* iI_xx, *iI_yy, *iI_zz;          // body-frame inverse inertia diagonal
float* iI_xy, *iI_xz, *iI_yz;          // off-diagonal (symmetric)
uint32_t* shape, *flags;
uint32_t n;
```

SoA enables coalesced memory access across work-items (adjacent bodies touch adjacent memory addresses) and is a prerequisite for future automatic differentiation (persistent across frames, no re-allocation).

### 4.2 Integrator

Semi-implicit (symplectic) Euler with substep count as a runtime parameter. Velocity is updated from forces first, then position from velocity — the standard choice for GPU physics because it preserves energy better than explicit Euler with the same step size and requires no solve to step forward.

### 4.3 Broadphase — Parallel LBVH

Linear Bounding Volume Hierarchy (Karras 2012) built and refitted entirely on the GPU each frame.

**Build steps** (per frame):
1. Compute AABB per body (from shape half-extents + pose).
2. Compute 10-bit Morton code per body (scene AABB quantised to 2^10 cells per axis).
3. Sort by 64-bit key `(morton30 << 32) | body_idx` using the bitonic sorter. Embedding the body index as a tiebreaker makes the sort **stable**, producing a deterministic tree structure even when multiple bodies share a Morton cell.
4. Build internal nodes using Karras' parallel radix-tree algorithm (each internal node assigned to the longest common prefix split between adjacent sorted leaves).
5. Refit bottom-up using Karras' atomic-flag pattern: each leaf thread walks its path to the root, acquiring internal nodes via an atomic counter. The second thread to arrive at a node computes the merged AABB and continues upward.
6. Traverse: each leaf queries the BVH to collect overlapping candidate pairs `(i, j)` with `i < j`. Pairs are accumulated into a flat buffer using `atomic_add_seq`.
7. Sort pairs by canonical `(a, b)` key (bitonic sort) to produce a deterministic ordered list for the narrowphase.

**AABBs are stored in fp16** to halve traversal bandwidth, and the six bounds of a node are **packed into one 16-byte cache-line-aligned record** (`NodeBox`). Traversal visits nodes in tree order — effectively random in memory — so co-locating a node's bounds turns the per-node fetch from six scattered fp16 loads into a single 128-bit load. This is the dominant cost in both the pair query and ray-query traversal (see §9.4). False positives from fp16 rounding are handled by the authoritative narrowphase and do not affect correctness.

### 4.4 Narrowphase

Sequential processing over the deterministic sorted pair list for reproducibility. Supports:

- **Sphere–sphere**: distance vs. sum of radii.
- **Sphere–box**: closest point on box surface to sphere centre.
- **Box–box**: 15-axis SAT (3 × 2 face normals + 9 edge–edge cross-products). Up to 4 contact points via face-area heuristic: the larger body's face is always the reference, so small-box-on-large-ground always generates contacts.
- **Sphere–TriangleMesh**: sphere centre to closest point on triangle.
- **Box–TriangleMesh**: 13-axis SAT.
- **ConvexHull–ConvexHull**: GJK (simplex-based) → EPA (expanding polytope). Fixed-size buffers (64 vertices, 128 faces, 32 iterations max) to avoid dynamic allocation in device code.

### 4.5 Solver — XPBD

Extended Position-Based Dynamics (Müller et al. 2020). Gauss-Seidel over contacts and joints, parallelised on the GPU by **graph coloring** (§4.5.1).

**Why XPBD over impulse-based LCP**: XPBD is GPU-natural (no global constraint matrix), naturally handles contact + joint constraints in a unified loop, and is simpler to implement correctly solo than a projected-Gauss-Seidel LCP solver. The tradeoff is that XPBD is not fully physically accurate for high-stiffness joints, but it is sufficient for manipulation simulation.

**Articulations use XPBD joint constraints, not Featherstone**: Fixed, Revolute, Prismatic, and Ball joints are solved as XPBD position/orientation constraints interleaved with contact solving. Featherstone is an O(n) optimisation for long unbranched chains (100+ links); a 7-DOF Franka arm does not justify the added complexity, and the unified constraint infrastructure keeps the solver loop architecturally simple.

**Contact constraint**:
```
generalized_inv_mass = inv_mass_A + inv_mass_B
    + (r_A × n) · (I_A_world⁻¹ (r_A × n))
    + (r_B × n) · (I_B_world⁻¹ (r_B × n))
delta_lambda = (-C - alpha_tilde * lambda) / (w + alpha_tilde)
```
Full angular response: `I_world⁻¹ = R · I_body⁻¹ · Rᵀ`. Quaternions are re-normalised after each Gauss-Seidel iteration to prevent drift accumulation.

**Joint types**: Fixed, Revolute (with limits and PD motor), Prismatic, Ball. PD motor: a spring-damper velocity correction applied along the joint axis proportional to position error and velocity error, parameterised by `stiffness` and `damping`.

**Sleeping / energy dissipation**: zero-restitution velocity correction zeroes the normal relative velocity at contact (velocity-level constraint). Per-contact accumulated lambda prevents multi-iteration over-correction.

#### 4.5.1 Parallelism — graph coloring

The solve was originally a single GPU work-item processing all contacts sequentially (chosen for deterministic Gauss-Seidel ordering). At scale this became the dominant cost — ~3 765 contacts × 10 iterations on one of ~3 500 GPU cores — and was why the GPU lost to the CPU at every scene size.

It is now parallelised by **graph coloring**, the approach used by NVIDIA Newton/Warp (`warp.sim.graph_coloring`) and PhysX 5 TGS. Constraints (contacts + joints) are partitioned into colours such that no two constraints in a colour share a *dynamic* body. Within a colour, work-items touch disjoint bodies, so the colour solves in one parallel kernel with **no atomics**; colours are dispatched in queue order, preserving Gauss-Seidel coupling *across* colours. This is **deterministic by construction** — fixed colour order, disjoint writes — so it satisfies the determinism gate, unlike a Jacobi/atomic sweep.

Key design points:
- **Static/kinematic bodies (`inv_mass == 0`) are not coloring nodes.** Their writes are guarded no-ops (`w·Δλ == 0`), so they never conflict; this keeps all 1 024 box↔ground contacts in a handful of colours instead of forcing ~1 024. (PhysX/Newton do exactly this.)
- **Coloring** is a single-work-item greedy pass keyed on constraint index: each constraint takes the lowest colour not yet used by either of its dynamic bodies (a per-body 64-bit used-colour mask; spill → serial fallback if > 64 colours). Run once per frame, amortised over `n_iters` parallel sweeps.
- **Quaternion renormalisation** moved to its own per-body parallel kernel (was per-iteration inside the serial kernel).
- **Serial fallback** retained for scenes ≤ `kSerialThreshold` (256 constraints) — keeps small scenes / goldens unchanged — and as a determinism oracle (`set_force_serial`) cross-checked against the colored path in `xpbd_solver_test`.

Result: `stress_test_blocks` (1 025 bodies) went ~101 → **~237 fps warm** on the RTX 3060, and the GPU now beats the OpenMP CPU (~205 fps) for the first time. Result hashes changed (colour order ≠ contact-index order); goldens were regenerated.

### 4.6 Determinism

Determinism on the same hardware + build is a hard requirement, enforced by CI.

Achieved by:
1. **Stable Morton sort**: 64-bit key `(morton30 << 32) | body_idx` makes equal-Morton bodies sort by index → deterministic, distinct keys for the Karras radix tree every run.
2. **Bitonic sort of broadphase pairs** by canonical `(a, b)` key → deterministic contact traversal order regardless of the order pairs were atomically appended.
3. **Narrowphase** parallelised via scratch-append + key-sort + gather, which reproduces the sequential pair/contact order (so the contact set and order are run-invariant).
4. **XPBD solve** parallelised by **graph coloring** (§4.5.1): fixed colour order + disjoint per-colour writes ⇒ deterministic by construction, no atomics.
5. **Level-synchronised BVH refit**: each merge round is a separate queue-ordered kernel, so a parent only ever reads child AABBs finalised by a *previous* kernel — no intra-kernel cross-thread read of in-flight data (this closed the former parallel-refit race, see below).

The CI determinism gate (`test_core_sim_determinism`) covers an 8-box scene, a 2-link arm, **and a 256-box stacked scene** (the large case exercises the colored solver and the deep BVH refit — the small scenes are serial/shallow and would not catch a coloring or refit race). It passes bit-identically on both backends, every run, and `stress_test_blocks` (1 025 bodies) now returns an identical hash across runs on CUDA (`3b31b708569886eb`).

> **Resolved — parallel BVH refit (was a CUDA-only race).** The Phase 5.5 parallel refit used a Karras atomic-flag rendezvous and read a node's two child AABBs once an atomic counter signalled both subtrees done; on CUDA the device-scope atomic did not reliably order the non-atomic fp16 AABB writes against the flag, so a deep tree could read a stale child AABB and diverge run to run. It is replaced (2026-06-03) by a **level-synchronised** refit: a parallel leaf-AABB pass, then merge rounds each dispatched as a *separate* queue-ordered kernel (a node merges once both children are finalised; `d_flags_` is the per-node ready flag, leaves implicitly ready). Parents read only previously-finalised children ⇒ bit-identical every run, still parallel (~237 fps vs ~177 for a single-work-item fallback). Rounds run in batches of 16, stopping once the root is finalised (Karras tree height ≤ 64 for distinct 64-bit keys).

CPU and GPU hashes differ by design (different FP unit behaviour, FTZ on GPU). Same-backend, same-build determinism is the requirement.

**Golden hashes** (Debug, physics-level determinism tests, current):
- 8 boxes + ground, 20 frames — OMP `0xcbcd209c3665c818`, CUDA `0xfb43a5a0352e1fe8`
- 2-link revolute arm, 20 frames — OMP `0x44d086af21f338cf`, CUDA `0x751ebe6959a2507c`
- 256-box stack (colored solver), 20 frames — run-to-run equality only (no fixed golden)

---

## 5. Scene & Asset Pipeline

### 5.1 SDF/URDF Loading

libsdformat processes SDF and URDF (via internal URDF→SDF translation). Each SDF link becomes exactly one `BodyDesc`; joint anchors are resolved in the parent link's frame via `SemanticPose().Resolve()`. The native SDF coordinate convention (z-up, gravity `{0, 0, -9.81}`) is preserved without coordinate transformation.

### 5.2 Convex Decomposition

V-HACD v4.1.0 decomposes non-convex meshes into approximate convex hulls at scene load time (CPU, one-off). Each hull is uploaded to `ConvexHullStore`, a flat SoA buffer of vertices with per-hull start+count offsets.

### 5.3 Triangle-Mesh BVH

For static geometry, a median-split BVH is built on the CPU and uploaded once. Node layout: `{min_x, min_y, min_z, max_x, max_y, max_z, left, right, tri_idx}`. Leaves have `left < 0`. GPU traversal uses a fixed 64-entry stack (no dynamic allocation).

`MeshBvhStore` merges per-mesh node arrays into a single flat buffer; child indices are offset at `add()` time so traversal only needs the flat array + per-mesh root index.

---

## 6. Sensor API

### 6.1 Ray Queries

`ray_query(BvhView bvh, Ray ray, float t_max)` — batched BVH traversal on the GPU, returning the closest hit `{body_idx, t, normal}`. Uses fp16 AABB slab tests for the broadphase traversal, followed by exact intersection for leaves (quadratic for spheres, OBB for boxes). The same BVH used for broadphase collision detection serves ray queries — no second BVH required.

### 6.2 Contact Sensors

`ContactSensor` provides per-shape contact event subscription. After narrowphase, `query()` downloads the narrowphase output and filters contacts for subscribed shapes on the host. Python-accessible with zero-copy NumPy views (OWNDATA=False, wrapping the device buffer directly via the CUDA array interface / DLPack protocol).

---

## 7. Python Bindings

nanobind module exposing:
- `Device`, `Stream` — context management
- `BodyStore`, `ShapeStore` — scene population
- `Broadphase`, `Narrowphase` — physics pipeline
- `ContactSensor` — sensor data access with NumPy integration

nanobind was chosen over pybind11 for its lower overhead and tighter C++20 integration. The module is built as a shared library (`.so`) with `CMAKE_POSITION_INDEPENDENT_CODE=ON`.

---

## 8. Visualization Tools (Phase 5.5)

The visualization tool in `tools/viz/` is a Magnum-based standalone executable with three modes. It is not on the CI path and is never invoked automatically — it is a human inspection tool only.

### Scene descriptor format

Every demo now writes a `{prefix}.scene` binary alongside its `{prefix}.trajectory`. Format:

```
Header (8 bytes): uint32_t n_bodies, n_shapes
Per-body (4 bytes): uint32_t shape_idx
Per-shape (20 bytes): uint32_t type, float half_x, half_y, half_z, uint32_t _pad
```

This enables the visualiser to reconstruct box and sphere geometry from trajectory pose data.

### Operating modes

- `viz snapshot <prefix> [--frame N] [-o out.png]` — headless EGL rendering (no display required), outputs a PNG via Magnum's `StbImageConverter`. Intended for whitepaper figures.
- `viz replay <prefix> [--fps N] [--loop]` — GLFW interactive window. Orbit camera (left-drag, scroll zoom), spacebar pause, arrow-key step, Q/Esc quit.
- `viz live` — stub (shared-memory ring buffer consumer, not yet implemented). The producer side is `VizSink` in `core/include/core/viz_sink.hpp` — a 30-LoC POSIX shm ring buffer that the simulation can call each frame.

Magnum was chosen over raw OpenGL (~1 200 LoC self-hosted) because its vcpkg port at the pinned baseline provides EGL headless (`WindowlessEglApplication`), GLFW interactive (`GlfwApplication`), built-in box and sphere primitives (`Primitives::cubeSolid`, `Primitives::uvSphereSolid`), and a Blinn-Phong shader (`Shaders::Phong`) — reducing self-hosted LoC to ~450 while remaining Apache-2.0 licensed.

---

## 9. Performance & Test Results

All figures below are measured on the reference hardware: **NVIDIA GeForce RTX 3060** (sm_86) for CUDA and **Intel Xeon E5-2667 v4 @ 3.20 GHz, 4 OpenMP threads** for the CPU backend. Demos are Release (`-O3`); GPU figures are **warm-cache** — AdaptiveCpp JIT-compiles kernels on the first run, so a cold first run is ~1.5–2× slower (e.g. stress_test_blocks CUDA: ~36 fps cold → ~55 fps warm). The "det." column is run-to-run hash stability over 3 runs on that backend.

### 9.1 Demo performance

**manipulator_pick** — Franka Panda contact-friction pick-and-place (v0.2/v0.3 demo).
16 bodies: Panda base + 7 links + hand + 2 fingers (loaded from URDF, driven kinematically), plus a ground plane, table, pickup cube, and 2-cube tower (dynamic). Full broadphase + narrowphase + XPBD, 16 substeps/frame, μ = 5 contact friction. 800 frames (13.3 s). The cube is held by **finger friction only** (no fixed joint), lifted, transported, and dropped onto the tower.

| Backend | FPS | Realtime | Grasp OK | Run-to-run det. |
|---|---|---|---|---|
| CUDA (RTX 3060) | 159.7 | 2.66× | yes | yes (`a0230d602c378203`) |
| OpenMP (4 threads) | 260.3 | 4.34× | yes | yes (`1f0d4116979b0ee9`) |

**stress_test_blocks** — 1 025 rigid bodies (1 024 dynamic 0.4 m boxes stacked 8×8×16 + ground), ~3 400 contacts/frame, 300 frames (5 s).

| Backend | FPS | Realtime | Run-to-run det. |
|---|---|---|---|
| CUDA (RTX 3060) | **~233 warm** / ~84 cold | 3.9× | yes (`3b31b708569886eb`) |
| OpenMP (4 threads) | 205.5 | 3.42× | yes (`6ffbaef0a15d3e13`) |

**arm_push** — SDF-loaded 1-DOF arm sweeping into 2 pushable boxes (5 bodies, 1 revolute joint, full contact pipeline), 600 frames.

| Backend | FPS | Realtime | Run-to-run det. |
|---|---|---|---|
| CUDA (RTX 3060) | 170.4 | 2.84× | yes (`0x6030f7fd8a445e92`) |
| OpenMP (4 threads) | 5 473 | 91.2× | yes (`0xaac0fc87da9672b9`) |

The GPU now **outperforms the CPU at the stress scale** (1 025 bodies): ~233 fps CUDA vs ~205 fps OpenMP, the first scene where the GPU wins, after parallelising the narrowphase (§4.4) and the XPBD solver (§4.5.1) and removing the BVH-refit serialisation (§4.6). At small scene sizes (manipulator_pick, arm_push: ≤ 16 bodies) OpenMP still wins — GPU per-frame overhead (kernel-launch latency plus the single `download_count` sync per frame) dominates the tiny per-frame compute there. The crossover moves further in the GPU's favour as body count grows. (Hashes differ between backends by design — different FP unit behaviour; only same-backend, same-build run-to-run stability is a requirement.)

### 9.2 Test suite

`ctest` over the `smoke` + `determinism` labels, Debug build:

| Backend | Result | Wall time |
|---|---|---|
| CUDA (RTX 3060) | **15 / 16 pass** | 37.3 s |
| OpenMP (4 threads) | **16 / 16 pass** | 17.8 s |

The single CUDA failure is `test_core_math_equiv` (host-vs-device math parity): one element of one transcendental result differs by 1 ULP on the GPU. It passes on OpenMP and does not affect physics determinism. (`sim_determinism` now has three cases — 8 boxes, 256-box stack, 2-link arm — the 256-box case gating the colored solver + deep refit.) Per-test timings (CUDA, seconds):

| Test | s | Test | s |
|---|---|---|---|
| compute_smoke | 1.9 | core_xpbd | 2.3 |
| compute_determinism | 0.7 | sim_determinism | 0.2 |
| core_math_equiv | 0.96 ✗ | ray_query | 3.1 |
| core_articulation | 1.0 | scene_smoke | 14.1 |
| core_body | 0.1 | arm_push_smoke | 5.7 |
| core_integrator | 0.4 | pytest_contact_sensor | 0.5 |
| core_broadphase | 5.1 | core_narrowphase | 1.4 |

`scene_smoke` dominates because it runs the V-HACD convex-decomposition pipeline end to end.

### 9.3 Determinism status

Same-backend, run-to-run determinism now holds **on both backends at all tested scene sizes** — manipulator_pick, arm_push, the three CI golden-gate scenes (§4.6), and `stress_test_blocks` (1 025 bodies). The CI determinism gate (`test_core_sim_determinism`: 8 boxes + 256-box stack + 2-link arm) passes bit-identically on both backends on every run.

**`stress_test_blocks` is now deterministic on CUDA at every scale tested** — 1 025 bodies (`3b31b708569886eb`), 4 097 bodies (`ba0aa6b74164e0ac`), and 8 193 bodies (`d31696ce471ce816`), each identical across runs — closing the former large-scene non-determinism. The root cause was the parallel BVH refit's cross-thread read of child AABBs ordered only by a device-scope atomic; it is fixed by the level-synchronised refit (§4.6), where each merge round is a separate queue-ordered kernel so parents read only previously-finalised children. The new 256-box determinism case in the CI gate exercises the deep-BVH / colored-solver path that the small golden scenes did not.

### 9.4 Scaling — 1k / 4k / 8k bodies

`stress_test_blocks` was profiled on CUDA at three grid sizes via `--grid NX NY NZ --profile`, which times each pipeline stage behind an `s.wait()` barrier (averaged over 200 frames after 30 warm-up frames). The barriers make each stage's wall time its true GPU cost but serialise the pipeline, so the stage-summed fps is a *lower bound*; the "pipelined warm" row is the actual end-to-end demo throughput (no per-stage barriers).

| Stage | 1k (8×16×8) | 4k (16×16×16) | 8k (16×32×16) |
|---|---|---|---|
| integrate | 0.018 ms | 0.021 ms | 0.022 ms |
| build_and_query | 0.92 ms (10.9 %) | 2.06 ms (19.5 %) | 2.97 ms (18.4 %) |
| download_count | 0.018 ms | 0.024 ms | 0.025 ms |
| sort_pairs | 0.53 ms (6.2 %) | 0.87 ms (8.2 %) | 1.17 ms (7.2 %) |
| narrowphase | 0.51 ms (6.0 %) | 0.86 ms (8.1 %) | 0.95 ms (5.9 %) |
| **solve (XPBD, colored)** | **6.48 ms (76.5 %)** | **6.72 ms (63.7 %)** | **11.05 ms (68.2 %)** |
| **total (stage-summed)** | 8.47 ms | 10.56 ms | 16.20 ms |
| stage-summed fps (lower bound) | 118 | 95 | 62 |
| **pipelined warm fps** | **~235** | **~111** | **~78** |
| contacts/frame (mid-settle) | ~3 700 | ~19 400 | ~36 700 |
| candidate pairs/frame | ~7 500 | ~40 000 | ~85 000 |

**The GPU holds above realtime (≥ 60 fps) through 8 192 bodies.** Findings:

- **The colored solver dominates (64–77 %) but scales *sub-linearly*.** Solve is nearly flat 1k→4k (6.48 → 6.72 ms) despite 5× the contacts, then grows by 8k. The colored solve dispatches `n_iters × n_colors` kernels per frame, and `n_colors` is roughly constant for a stacked grid (each box contacts a bounded number of neighbours, so the conflict graph's chromatic number does not grow with body count). It is therefore **kernel-launch-bound** through 4k and only becomes compute-bound by 8k — the opposite of the old single-work-item solver, which scaled linearly with contact count. This flat region is exactly where GPU parallelism pays off.
- **`build_and_query` (BVH build + level-synchronised refit + traversal) is the clear #2 cost** and grows roughly linearly with body count. It was attacked next (see below): a sub-stage profile showed the **top-down traversal is ~70 % of it** and is memory-bound, and packing the node AABBs cut `build_and_query` by **~9–15 %** (4k 2.06 → 1.76 ms, 8k 2.69 → 2.45 ms).
- **`sort_pairs` is *not* the bottleneck** (6–8 %), even at 85 000 pairs where it falls back to the multi-launch bitonic path (Issue 2a, §13). That known issue is far less urgent in practice than its historical 77 %-of-frame profile suggested — the broadphase work since (single-kernel sort for n ≤ 1024, fp16 AABBs) and the now-parallel solver moved the bottleneck elsewhere.
- The capacity-sensitive buffers (broadphase pairs, narrowphase contacts, solver lambda accumulator) scale with body count; the solver's `lambda_c_` must be sized to the maximum contact count (a default 16 384 overflows past ~16k contacts).

**Attacking `build_and_query` (the #2 cost).** A per-step sub-profile (each of the seven build steps fenced) at 8 192 bodies put the cost almost entirely in the **top-down pair-query traversal (~70 %)**, with the Morton sort second (~18 %) and the refit third (~9 %); Morton/Karras/root steps are negligible. The traversal is **memory-latency-bound**, not control-flow- or atomic-bound — confirmed by two negative experiments: a *stackless rope (escape-pointer) traversal* (removing the per-thread 64-entry stack) gave **no speed-up**, and switching the per-pair counter from a `seq_cst` to a relaxed atomic also gave none. The cost is the per-node AABB fetch: traversal visits nodes in tree order (random in memory) and each visit read six fp16 bounds from **six separate arrays** — up to six scattered cache-line loads per node. Packing the bounds into one 16-byte `NodeBox` (§4.3) makes each visit a single 128-bit load, cutting `build_and_query` ~9–15 % and speeding ray-query traversal for free, with **bit-identical results** (same fp16 values, only the layout changed — the 1 025-body hash is unchanged). The remaining traversal cost is the inherent per-leaf descent from the root; further gains need an algorithmic change (e.g. a BVTT front), deferred.

### 9.5 Optimisation history — the migrating bottleneck

GPU performance was reached by a sequence of targeted passes, each of which moved the dominant cost to the next stage rather than uniformly speeding everything up. Recording the progression matters because it shows *where* the bottleneck lived at each step and why the next change was the right one — the same single-frame profile that once read "77 % sort" now reads "68 % solver."

| Stage of the work | Dominant cost (CUDA) | Change made | Result |
|---|---|---|---|
| Phase 5.5 (512 bodies) | broadphase `sort_pairs` — **77 %** of frame | Single-work-group bitonic sort (all bitonic passes in one kernel for n ≤ 1024) + fp16 BVH AABBs | sort 4.26 ms → ~0.1 ms; sort drops to ~3 % |
| Re-profile (1 025 bodies) | **narrowphase 47 % + solver 46 %** — both single-work-item kernels | — (diagnosis) | bottleneck split between the two remaining serial passes |
| Narrowphase parallelised | narrowphase ~9.4 ms (single work-item) | One work-item per candidate pair → scratch-append → key-sort → gather (reproduces sequential contact order, so determinism holds) | ~9.4 → ~1 ms; **~55 → ~101 fps** warm |
| Solver parallelised + refit fixed | **XPBD solver ~9 ms** (single work-item, ~46 %) | Graph-colored parallel solve (§4.5.1) + level-synchronised deterministic BVH refit (§4.6) | **~101 → ~233 fps** warm; GPU beats the CPU for the first time; CUDA determinism restored at scale |
| Multi-scale profile (§9.4) | colored solver 64–77 %, but **launch-bound and near-flat** through 4k | — (diagnosis) | next target is `build_and_query` (BVH build/refit/traversal), which now scales ~linearly; `sort_pairs` is no longer material |
| `build_and_query` traversal | top-down pair query **~70 %** of `build_and_query`, **memory-latency-bound** (rope traversal and relaxed atomics both gave nothing) | Pack the six fp16 node bounds into one 16-byte `NodeBox` ⇒ one 128-bit load per node visit | `build_and_query` −9–15 %; ray-query faster for free; bit-identical results. Also fixed a latent bug: the Morton sort's GPU fast path was gated by a compile-time macro, so the CUDA build silently ran the barrier-broken fast path on `Device::default_cpu()` (the determinism gate) — now gated at runtime by device type |

Two structural lessons hold across the sequence. First, **every "fix" relocates the bottleneck** — there is no single hot spot, only a current one — so profiling has to be repeated after each change rather than trusted from a stale table, and a hypothesis about *why* a stage is slow must be confirmed before optimising it (the traversal looked stack/atomic-bound but was neither). Second, **determinism was preserved at each step, not bolted on afterward**: the parallel narrowphase reproduces the sequential contact order via key-sort, the colored solver is deterministic by construction (fixed colour order, disjoint writes), and the level-synchronised refit reads only previously-finalised data — so the CI determinism gate held green through the entire optimisation.

---

## 10. Output Format

Each demo executable emits three or four files per run:

| File | Content |
|---|---|
| `{prefix}.trajectory` | Binary: `uint32_t n_bodies, n_frames`; then per-frame, per-body `float[7] = {x, y, z, qw, qx, qy, qz}`, sampled every other simulation frame (30 Hz) |
| `{prefix}.scene` | Binary: shape catalog mapping body indices to geometry (type + half-extents). Consumed by `viz` |
| `{prefix}.metrics.json` | Wall-clock time, fps, realtime factor, average contacts per frame |
| `{prefix}.golden` | FNV-1a-64 hex hash of final body state (positions + orientations of all bodies) |

The FNV-1a-64 hash (`prime = 0x00000100000001B3`, `offset = 0xcbf29ce484222325`) covers positions and orientations of all bodies. A mismatch on any CI run fails the determinism gate.

---

## 11. Testing Infrastructure

Tests use Catch2 v3 (includes microbenchmark support via `BENCHMARK(...)`). Two test categories are registered via CMake helpers:

- `dyphur_add_smoke_test(target)` — label `smoke`. Fast correctness checks run on every build, covering compute, math, body dynamics, articulations, integrator, broadphase, narrowphase, XPBD, ray queries, scene loading, arm_push, and Python bindings.
- `dyphur_add_determinism_test(target)` — label `determinism`. Golden-hash regression: same input → same output.

The combined `smoke` + `determinism` set is **15 tests**. Measured results (pass counts, wall time, per-test timings) are in §9.2; current status is 15/15 on OpenMP and 14/15 on CUDA (the one failure is a 1-ULP host/device math-parity check, §9.2).

**CI workflow presets**:
- `ci-linux-cuda` — CUDA backend, smoke tests
- `ci-linux-hip` — HIP/ROCm backend, smoke tests
- `ci-linux-cpu` — OpenMP CPU backend, full test matrix + sanitizers (ASan, UBSan, TSan)
- `ci-determinism` — Release build, golden-hash gate

All tests run headless. No display, no window, no graphics context required. Visualization is a separate, manually-invoked step.

---

## 12. Key Design Decisions — Rationale Summary

| Decision | Rationale |
|---|---|
| **SYCL 2020 / AdaptiveCpp** | Only production-ready standard targeting CUDA + HIP + CPU in a single source base. Avoids maintaining three backend-specific codebases. |
| **SoA data layout** | SIMD- and GPU-cache-friendly; coalesced memory access across work-items; prerequisite for automatic differentiation (persistent allocations, no reallocation). |
| **XPBD solver** | GPU-natural (no global matrix); unified contact + joint constraint loop; simpler implementation than LCP for solo development. Accurate enough for manipulation. |
| **XPBD joints (not Featherstone)** | Featherstone is O(n) for unbranched chains — a benefit only at 100+ links. 7-DOF arms don't justify the added complexity. Unified constraint loop is architecturally cleaner. |
| **Parallel narrowphase + graph-colored solver** | Determinism *with* full constraint-level GPU parallelism: the narrowphase reproduces sequential contact order via key-sort + gather, and the solver uses graph coloring (fixed colour order, disjoint per-colour writes ⇒ deterministic, atomic-free) — the approach used by NVIDIA Newton/Warp and PhysX 5 TGS. Replaced the original single-work-item passes once they became the dominant GPU cost. |
| **Determinism as a CI gate** | Physics bugs that only appear stochastically are the hardest to debug. Bit-identical determinism catches them with a 1-line test. ~10–30% perf cost accepted. |
| **vcpkg manifest mode, baseline-pinned** | Reproducible builds without per-developer environment management. AdaptiveCpp excluded (requires system install with LLVM + GPU SDKs). |
| **Headless-first, no rendering** | Decouples correctness from display availability. CI runs identically to local development. Visualization is a separate tool consuming trajectory dumps, never a validation step. |
| **Hand-rolled device math** | Eigen's SYCL support is unofficial; it cannot be depended on solo. Bespoke SoA types give full layout control, mirror Eigen's API surface, and remain inside the kernel allow-list. |
| **Magnum for visualization** | ~450 LoC self-hosted vs ~1 200 for raw OpenGL. vcpkg port available at pinned baseline. EGL headless + GLFW interactive in one binary. Apache-2.0 license matches project. |
| **nanobind over pybind11** | Lower overhead per-call, tighter C++20 integration, zero-copy NumPy views via DLPack without extra wrapping. |

---

## 13. Known Limitations & Open Work

| Item | Description |
|---|---|
| Bitonic sort fast path is GPU-only | The single-kernel (work-group local-memory) bitonic sort is correct only on GPU backends; AdaptiveCpp's OpenMP backend mis-synchronises the work-group barriers, so the CPU build uses the multi-launch path. For pair counts > 1024 (scenes > ~1 000 bodies) **all** backends use multi-launch. Fix: oneDPL / CUB radix sort. This is the next GPU-scaling bottleneck now that the solver is parallel. |
| GPU underperforms CPU at small scenes | At ≤ ~16 bodies (manipulator_pick, arm_push) GPU launch + per-frame sync overhead dominates the tiny per-frame compute. The GPU now wins at the 1 025-body stress scale (~233 vs ~205 fps); the crossover moves further in the GPU's favour as body count grows. |
| Single GPU | Multi-GPU spatial decomposition is planned (Phase 4) but not started. |
| Windows | Deferred to v0.3+; current code is Linux-only. |
| Differentiability | Architecture is AD-ready (persistent SoA buffers, no re-allocation, kernels structured for adjoint formulation) but not yet implemented. |
| Live visualization | `viz live` mode stub only; `VizSink` POSIX shm producer is implemented. |
| Heightfield, SDF-volume collisions | Explicitly out of scope. |
| Soft bodies, cloth, fluids | Explicitly out of scope. |

---

## 14. Repository Layout

```
compute/include/compute/    Compute shim headers (device.hpp, buffer.hpp,
                              parallel_for, sort, reduce, atomics, stream)
core/include/core/          Physics headers (body, broadphase, narrowphase,
                              solver, joints, ray_query, contact_store,
                              shape_store, viz_sink)
core/include/core/math/     Device-safe math types (Vec3, Quat, Mat3,
                              Transform, SpatialVector, Inertia)
core/src/                   Physics implementations (broadphase.cpp,
                              narrowphase.cpp, xpbd_solver.cpp, …)
core/tests/                 Catch2 smoke + determinism tests
scene/include/scene/        SDF loader, scene graph, asset types
scene/src/                  sdf_loader.cpp, mesh_bvh.cpp, convex_hull_store.cpp
examples/common/            scene_io.hpp — .scene descriptor read/write
examples/stress_test_blocks/ 1k-body headless demo (v0.1 exit criterion)
examples/manipulator_pick/  Franka Panda (URDF) contact-friction pick-and-place;
                              assets/ holds the URDF + collision STL meshes
examples/arm_push/          SDF-loaded arm + boxes with contacts (v0.3)
tools/viz/                  Magnum trajectory visualiser (replay + snapshot)
bindings/python/            nanobind Python module
cmake/                      BackendOptions, DependencyAllowlist, DeterminismCI
docs/PLAN.md                Full design rationale and phased roadmap
docs/known_issues.md        Documented issues with root cause and fix options
```

---

## 15. Dependencies

| Library | Version / Source | Role |
|---|---|---|
| AdaptiveCpp | System install (LLVM + CUDA/ROCm) | SYCL 2020 compute runtime |
| Eigen3 | vcpkg | Host-side linear algebra; robotics interop |
| spdlog | vcpkg | Structured logging |
| fmt | vcpkg | String formatting |
| Catch2 3.4+ | vcpkg | Tests and microbenchmarks |
| pugixml | vcpkg | XML parsing (SDF/URDF support) |
| libsdformat | vcpkg | SDF/URDF scene loading |
| V-HACD 4.1 | vcpkg | Convex decomposition of non-convex meshes |
| Tracy | vcpkg | CPU + GPU profiling (NVTX/rocTX markers) |
| nanobind | pip / `.venv` | Python bindings (vcpkg nanobind broken at pinned baseline) |
| Magnum 2020.06 | vcpkg | Visualization (EGL headless + GLFW interactive) |
| magnum-plugins | vcpkg | PNG output via StbImageConverter |

All dependencies except AdaptiveCpp are managed by vcpkg with a pinned baseline (`aa40adda5352e87655b8583cfb2451d5e9e276fd`).
