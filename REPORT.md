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
- `sort_by_key(Stream&, Key*, Value*, size_t n)` — bitonic sort. For `n ≤ 1024`: single kernel launch using SYCL local memory — all `O(log² n)` bitonic passes execute on-chip without PCIe round-trips. For `n > 1024`: multi-launch fallback.
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

**AABBs are stored in fp16** to halve traversal bandwidth. False positives from fp16 rounding are handled by the authoritative narrowphase and do not affect correctness.

### 4.4 Narrowphase

Sequential processing over the deterministic sorted pair list for reproducibility. Supports:

- **Sphere–sphere**: distance vs. sum of radii.
- **Sphere–box**: closest point on box surface to sphere centre.
- **Box–box**: 15-axis SAT (3 × 2 face normals + 9 edge–edge cross-products). Up to 4 contact points via face-area heuristic: the larger body's face is always the reference, so small-box-on-large-ground always generates contacts.
- **Sphere–TriangleMesh**: sphere centre to closest point on triangle.
- **Box–TriangleMesh**: 13-axis SAT.
- **ConvexHull–ConvexHull**: GJK (simplex-based) → EPA (expanding polytope). Fixed-size buffers (64 vertices, 128 faces, 32 iterations max) to avoid dynamic allocation in device code.

### 4.5 Solver — XPBD

Extended Position-Based Dynamics (Müller et al. 2020). Sequential Gauss-Seidel over contacts, single GPU work-item for determinism.

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

### 4.6 Determinism

Determinism on the same hardware + build is a hard requirement, enforced by CI.

Achieved by five mechanisms:
1. **Stable Morton sort**: 64-bit key `(morton30 << 32) | body_idx` makes equal-Morton bodies sort by index → deterministic tree every run.
2. **Single-work-item AABB refit**: avoids non-deterministic memory-order dependence of GPU atomic AABB merges. (Empirical testing showed that `memory_scope::device` acquire-release atomics are insufficient for non-atomic AABB reads/writes under CUDA's actual memory model.)
3. **Bitonic sort of broadphase pairs** by canonical `(a, b)` key → deterministic contact traversal order.
4. **Single-work-item narrowphase** over sorted pairs.
5. **Single-work-item XPBD** with fixed contact traversal order.

CPU and GPU hashes differ (expected: different FP unit behaviour, FTZ mode on GPU). Same-backend determinism: identical hash every run.

**Golden hashes** (OMP/CPU, Release, physics-level determinism tests):
- Rigid-body scene (8 boxes + ground, 20 frames, 60 Hz): `0xcbcd209c3665c818`
- Articulated arm (2-link revolute, 20 frames, 60 Hz): `0x44d086af21f338cf`

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

## 9. Performance Results

All timings are wall-clock from the demo executables, Release build, single GPU or single CPU socket.

### 9.1 stress_test_blocks — 1 025 rigid bodies

Scene: 1 024 dynamic boxes (0.4 m half-extent, 1 kg, stacked 8×8×16) + 1 static ground plane (20 m × 0.5 m × 20 m). Simulated 300 frames at 60 Hz (5 s). Contact count: ~2 600/frame.

| Hardware | Backend | FPS | Realtime factor | Deterministic |
|---|---|---|---|---|
| Intel Xeon E5-2667 v4 @ 3.20 GHz | OpenMP (CPU) | **254** | **4.24×** | yes (`7168a504ac96d97e`) |
| NVIDIA GeForce RTX 3060 (sm_86) | CUDA / AdaptiveCpp SSCP | **83** | **1.39×** | yes (`db4db5b2b5c6a8e4`) |

The CPU outperforms the GPU at this body count because the GPU overhead (kernel launch latency, PCIe syncs per frame) exceeds the computation savings for 1k bodies. GPU advantage grows with scene size.

**GPU stage breakdown** (CUDA, 512 bodies, from `test_core_bench`):

| Stage | Mean latency | Share |
|---|---|---|
| Broadphase sort_pairs | 4.26 ms | 77% |
| Broadphase build+query | 0.90 ms | 16% |
| Integrator | 0.18 ms | 3% |
| Narrowphase | 0.12 ms | 2% |
| XPBD solver | 0.05 ms | 1% |

The sort dominates because the original multi-launch bitonic sort incurred ~77 µs per kernel launch × 45–55 launches per frame. This was the primary optimization target in Phase 5.5 performance work: a single-kernel bitonic sort (for pair counts ≤ 1024, covering practical body counts up to ~1 000) reduces the sort from O(log² n) launches to 1, expected ~40× sort speedup and ~3× overall frame time improvement for larger scenes.

**Performance improvements shipped (Phase 5.5)**:
- **Sort**: single work-group kernel with local memory for n ≤ 1024 — O(log² n) launches → 1 launch.
- **BVH refit**: changed from single-work-item sequential loop to Karras 2012 parallel refit (`parallel_for(n_leaves, ...)` with atomic-flag synchronisation). Expected: linear speedup in leaf count.
- **Sync removal**: eliminated one blocking `download_count` GPU sync per frame from the stress test hot loop.

### 9.2 manipulator_pick — 7-DOF arm

Scene: static base + 7 revolute-jointed links + pre-grasped block (9 bodies, 8 joints). PD-controlled scripted pick-and-place. No broadphase/narrowphase (joint-only).

| Hardware | Backend | Frames | FPS | Realtime factor |
|---|---|---|---|---|
| Intel Xeon E5-2667 v4 | OpenMP (CPU) | 600 | **~4 027** | **~67×** |

### 9.3 arm_push — SDF-loaded arm with contacts

Scene: 5 bodies (static floor + 2 pushable boxes + arm base + arm link), 1 revolute joint, loaded from `arm_room.sdf`. Full broadphase + narrowphase + XPBD per frame.

| Hardware | Backend | Frames | FPS | Realtime factor |
|---|---|---|---|---|
| Intel Xeon E5-2667 v4 | OpenMP (CPU) | 600 | **6 579** | **109.6×** |
| NVIDIA RTX 3060 | CUDA | 60 | 22 | 0.37× |

The CUDA figure for 5 bodies is sub-realtime because GPU overhead completely dominates; this demo is intended as a functional correctness gate, not a performance target.

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

- `dyphur_add_smoke_test(target)` — label `smoke`. Fast correctness checks run on every build. Currently 16 tests covering compute, math, body dynamics, articulations, integrator, broadphase, narrowphase, XPBD, ray queries, determinism, scene loading, arm_push, and Python bindings.
- `dyphur_add_determinism_test(target)` — label `determinism`. Golden-hash regression: same input → same output.

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
| **Sequential narrowphase + solver** | Determinism without sorting output of parallel passes. Straightforward correctness reasoning. GPU parallelism is expressed at the body/pair level, not the constraint level. |
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
| Bitonic sort n > 1024 | For pair counts > 1024 (scenes with > ~1 000 bodies), the sort still uses multi-launch. Fix: oneDPL or CUB radix sort. Expected ~40× speedup on the sort step. |
| GPU underperforms CPU at 1k bodies | GPU launch overhead dominates at small body counts. Advantage expected above ~5 000 bodies. |
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
examples/manipulator_pick/  7-DOF arm scripted pick-place (v0.2 exit criterion)
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
