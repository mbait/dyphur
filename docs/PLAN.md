# Plan: GPU-First Rigid-Body Physics Framework for Robotics Simulation

## Context

We are designing a new physics simulation framework from scratch, targeted at robotics. The framework will be linked as a library by simulator programs that drive industrial manipulators, drones, and mobile robots whose constructions are expressible in the SDF/URDF formats Gazebo supports.

The motivating gap: existing engines either (a) lock into a single GPU vendor (NVIDIA Newton/Warp, Isaac), (b) are CPU-bound or weakly GPU-accelerated (Bullet, ODE, MuJoCo CPU paths), or (c) are not designed for the modern realtime-on-GPU regime that high-fidelity indoor robotics simulation now demands. We want GPU-first execution that is vendor-neutral, with a CPU fallback, deterministic on a given build+hardware combination, accurate enough for robotics, and at least realtime.

Constraints, gathered from prior conversation:

- Solo developer, full-time.
- Apache-2.0 license.
- Linux-primary; Windows added at late stage only if feasible. No mobile.
- C++ core with Python bindings (nanobind).
- Bit-identical determinism on same hardware + same build.
- All physics computation runs on GPU when available; CPU fallback otherwise.
- Reuse of third-party libraries is encouraged; the only prohibition is reusing entire physics engines or their solver components.
- Differentiability deferred; data layouts must not preclude adding it later.
- Sensors deferred; data layouts must not preclude adding them later.
- Rendering out of scope.

The ultimate destination is realistic simulation of indoor scenes with robots; the v0.1 milestone is a foundational rigid-body stress test that proves the GPU stack works end-to-end.

## Locked Design Decisions

| Concern | Decision |
|---|---|
| Implementation language | C++20 core + Python bindings via nanobind |
| Compute layer | AdaptiveCpp (SYCL 2020), backends: CUDA, HIP, OpenMP-CPU, optionally Level Zero |
| Build system | CMake (3.25+) with **Ninja Multi-Config** generator, `CMakePresets.json`, and **workflow presets** for every build/test scenario from day 1 |
| Package manager | vcpkg (manifest mode, `vcpkg.json`, baseline-pinned) |
| License | Apache-2.0 |
| Platform | Linux primary; Windows deferred to v0.3+ or later |
| Determinism | Bit-identical on same HW + same build; enforced via golden-trace CI gate |
| Solver philosophy | Solver-agnostic interface; XPBD chosen as the first/reference implementation (single algorithm, GPU-natural, simpler kernels for solo dev). Articulations use XPBD joint constraints, not Featherstone — consistent with contact solving, sufficient for ≤ 20-link chains. |
| Data layout | Structure-of-Arrays / AoSoA everywhere; persistent across frames (AD-ready) |
| Scene format | SDF/URDF via libsdformat (handles URDF→SDF internally) |
| Collision shapes v0.1–v0.2 | Box, sphere, capsule, cylinder, plane primitives; convex hulls via V-HACD; triangle mesh for static geometry. No heightfield, no SDF-volume. |
| Math (host) | Eigen — mandatory for robotics interop (Pinocchio, Drake, MoveIt, ROS all expose Eigen types) |
| Math (device) | Hand-rolled SoA/AoSoA-friendly types (`Vec3`, `Quat`, `Mat3`, `Transform`, `SpatialVector`, `Inertia`) — avoids depending on Eigen's unofficial SYCL story and gives full control of data layout. API surface mirrors Eigen idioms (`.cross()`, `.norm()`, `operator*`) to minimize cognitive load. |
| Differentiability | Not implemented; not precluded |
| Sensors | Not implemented; ray-query API planned for Phase 5 |
| Vulkan port | Deferred indefinitely; compute-shim interface designed not to preclude it |
| Development model | **Headless-first.** Dev and test happen in a headless environment with physical access to an NVIDIA RTX 3060 (Ampere, compute capability 8.6, 12 GB VRAM). Every check that runs in CI must also run locally headless. The *only* exception is end-to-end visualization, which is a separate, manually-invoked tool that consumes state dumps produced by headless simulation. |
| Automation philosophy | Maximally automated. No manual perf tuning, no "looks-fine-on-screen" verification. Every behavioral claim is backed by an automated test; every perf claim by an automated benchmark. Visualization is for human comprehension only, never for validation. |

## Architectural Overview

```
┌───────────────────────────────────────────────────────────────┐
│  Public API:  C++ headers  +  Python bindings (nanobind)      │
├───────────────────────────────────────────────────────────────┤
│  Scene / asset layer                                          │
│    SDF/URDF loader · convex decomp · BVH build · scene graph  │
├───────────────────────────────────────────────────────────────┤
│  Physics core   (solver-agnostic interfaces)                  │
│    Bodies · Articulations · Broadphase · Narrowphase ·        │
│    Contact manifold · Constraint graph · Integrator · Solver  │
├───────────────────────────────────────────────────────────────┤
│  Compute abstraction   (thin C++ layer over AdaptiveCpp/SYCL) │
│    Device · Buffer · Kernel · Reduction · Sort · Atomics      │
│    Backends: CUDA · HIP · Level Zero · OpenMP (CPU fallback)  │
├───────────────────────────────────────────────────────────────┤
│  Math + utilities                                             │
│    Host: Eigen · spdlog · fmt · pugixml · Catch2 v3 · Tracy  │
│    Device: hand-rolled SoA math (no Eigen on device)          │
└───────────────────────────────────────────────────────────────┘
```

Three structural rules that follow from the constraints and are non-negotiable:

1. **SoA / AoSoA everywhere in hot paths.** No AoS data crosses into a kernel.
2. **Solver is an interface.** The core ships with XPBD; alternate solvers must be addable without touching the rest of the physics core.
3. **Determinism is a CI gate, not a hope.** Every PR runs a golden-trace regression. The compute shim exposes deterministic-mode primitives (sorted reductions, fixed-iteration solvers) by default.

## Dependency Policy

Third-party C++ libraries fall into four roles. CI enforces the boundary.

1. **Build-time / load-time tools** (CPU, once per scene, produce GPU-ready buffers): libsdformat, V-HACD, pugixml, mesh loaders.
2. **Device-safe headers** (header-only, no allocation, no exceptions, no syscalls; usable inside kernels): our own SoA math types only. Eigen is explicitly excluded from device code — its SYCL support is unofficial and intermittent, and we cannot afford that risk solo. Lint-enforced allow-list in `compute/kernels/`.
3. **Host-side runtime helpers**: spdlog, fmt, Catch2 v3, nanobind. Never near kernels.
4. **Algorithms we reimplement on GPU**: GJK/EPA, LBVH construction, parallel scan/sort, contact-manifold clipping. Existing CPU implementations (libccd, Bullet) are correctness references, not binary dependencies.

Forbidden in kernels: STL containers, exceptions, virtual functions, dynamic allocation, file I/O.

## Automation & Headless Operation

This is a load-bearing constraint, not a nice-to-have. Concretely:

- **All tests, benchmarks, determinism checks, and demos run headless.** No code path requires a display, window, or graphics context to execute correctly.
- **Profiling tooling must produce offline artifacts.** Tracy is used in its headless-capture mode (server writes `.tracy` traces to disk, viewer is opened later on a workstation only when human inspection is wanted). NVTX / rocTX markers are emitted unconditionally; backend profilers consume them post-hoc.
- **Demos emit state dumps, not windows.** v0.1's "block pile / domino" demo runs as a headless executable that simulates for N seconds and produces:
  - A binary state-trajectory file (per-frame poses + velocities) for later visualization
  - A metrics JSON (steps/sec, bodies, contacts/frame, broadphase/narrowphase/solver/integrator timings, peak VRAM)
  - A determinism hash of the final state, checked against the golden value in CI
- **Visualization is a separate tool, manually run.** A standalone viewer (`tools/viz/`, Python + `rerun.io` or `meshcat`) reads the state-trajectory dump and replays it for human eyes. This tool is not part of the engine, not on the CI path, and not required for any automated check.
- **No "build it and look at it" loop.** Bugs are caught by tests, not by watching simulation runs. If a class of bug only surfaces visually, the response is to add an automated metric that quantifies it (e.g., "energy drift over 60s", "contact normal cosine distribution", "frame-time variance"), not to add a manual inspection step.
- **CI matrix runs entirely on headless runners.** Local development on the RTX 3060 happens via the same workflow presets; no per-environment divergence.

The single named exception: when a developer wants to see what's happening, they invoke the viewer tool against a saved trajectory. That is the only manual step in the entire loop.

## Phased Roadmap

### Phase 0 — Foundations
**Goal**: prove the compute abstraction end-to-end before any physics code lands.

- Repo skeleton: `core/`, `compute/`, `scene/`, `bindings/`, `tests/`, `examples/`, `cmake/`.
- CMake (3.25+) with **Ninja Multi-Config** as the only supported generator. `CMakePresets.json` at repo root defines:
  - **Configure presets** along two axes: backend (`cuda`, `hip`, `omp`) × sanitizer flavor (`clean`, `asan`, `ubsan`, `tsan`; sanitizers apply to `omp` only).
  - **Build presets** per config: `Debug`, `Release`, `RelWithDebInfo`.
  - **Test presets** for `smoke` (fast subset) and `full`.
  - **Workflow presets** that chain configure → build → test → bench: `dev` (Debug, smoke), `ci-linux-cuda`, `ci-linux-hip`, `ci-linux-cpu`, `ci-determinism` (Release + golden-trace gate only).
- vcpkg manifest (`vcpkg.json` at repo root, baseline-pinned). One triplet per platform; no per-config overrides.
- The single `dev` preset configures Debug for clangd/IDE consumption (resolves the multi-config `compile_commands.json` ambiguity).
- CI: each job invokes a single workflow preset. Linux + NVIDIA (`ci-linux-cuda`, mandatory), Linux + AMD (`ci-linux-hip`, should-pass), Linux + CPU-only (`ci-linux-cpu`, mandatory, sanitizer matrix), determinism gate (`ci-determinism`, runs on every PR).
- Compute shim: `Device`, `Buffer<T>`, `parallel_for`, `reduce`, `sort_by_key`, `atomic_add` (deterministic-mode flag), `Stream`, `Event`. Surface is the only thing the rest of the codebase calls — never raw SYCL.
- Math primitives: Eigen host-side; hand-rolled `Vec3`, `Quat`, `Mat3`, `Transform`, `SpatialVector`, `Inertia` device-side. Equivalence tests host-vs-device on shared inputs.
- Logging (spdlog), profiling (Tracy host + backend ranges), tests + benchmarks (Catch2 v3 — bundled microbenchmarks via `BENCHMARK(...)`, no separate benchmark dep).
- Determinism harness: golden-trace regression framework.

**Exit criterion**: saxpy + parallel reduction + parallel sort run deterministically and bit-identically on CUDA, HIP, and CPU.

### Phase 1 — Rigid Body MVP (v0.1 target)
**Goal**: free rigid bodies fall, collide, and settle stably at 1k+ bodies in realtime.

- Body data: SoA arrays for pose, velocity, inertia, mass, flags, shape handle. Persistent across frames.
- Integrator: semi-implicit (symplectic) Euler with substepping.
- Broadphase: parallel LBVH (Karras-style) built on GPU; refit per frame, rebuild on topology change.
- Narrowphase: analytical primitive-primitive pairs; GJK + EPA for general convex/convex; SAT for box/box.
- Contact manifold generation with clipping.
- Constraint solver interface (`IConstraintSolver`); first implementation: parallel XPBD on GPU.
- Sleeping / island detection via GPU parallel union-find.

**Exit criterion (v0.1)**: 1k+ rigid bodies in realtime on the RTX 3060 reference GPU, deterministic, on Linux + CUDA and Linux + CPU. Demo executable runs headless and emits a state-trajectory dump + metrics JSON + determinism hash. Visualization of the trajectory via the standalone viewer tool is a separate, manually-invoked step.

### Phase 2 — Articulations & Constraint Solver (v0.2 target)
**Goal**: drive a manipulator from a controller.

- Joint types implemented: Fixed, Revolute, Prismatic, Ball.
  SDF's "Continuous" is modelled as Revolute with limits disabled.
  Planar, Screw, and Gearbox are deferred to Phase 3+.
- **XPBD joint constraints** (not Featherstone): joints are solved as XPBD position/orientation
  constraints interleaved with contact solving in the existing sequential Gauss-Seidel loop.
  Featherstone is an O(n) optimisation for unbranched chains of 100+ links; Franka Panda's
  7-link chain does not justify the added implementation complexity, and XPBD joints share the
  same infrastructure as contact solving, keeping the codebase uniform.
- Coulomb friction in contact solver: global friction coefficient (default μ = 0.5);
  tangential velocity correction clamped by μ × normal impulse. Per-body friction deferred
  to Phase 3 when material properties are loaded from SDF.
- PD motor: revolute/prismatic joints with stiffness + damping > 0 get a velocity correction
  each solve iteration — a spring-damper impulse applied along the joint axis proportional
  to angle/position error and angular/linear velocity error.
- Joint limits for Revolute and Prismatic: one-sided angular/linear constraint applied when
  the joint angle/offset exits [limit_lo, limit_hi].
- **Solver interface extended**: `IConstraintSolver::solve()` now takes `const JointView&` as
  a third argument (between contacts and bodies). Existing call sites that have no joints
  pass `JointView{}` (zero-initialised, n=0 → joint loop is skipped).
- **Header split** (implemented in Task 1): `articulation.hpp` holds JointType/JointParams/JointView
  (no SYCL dependencies); `joint_store.hpp` holds JointStore (SYCL Buffer owner). Mirrors the
  body.hpp / body_store.hpp split so that pure-host smoke tests can include the type header.
- `examples/manipulator_pick`: 7-DOF Franka-like arm (base + 7 revolute links + pre-grasped
  block via Fixed joint). PD-controlled scripted 4-waypoint trajectory. Emits .trajectory,
  .metrics.json, .golden. URDF loading is Phase 3; hardcoding the robot validates Phase 2
  without the asset-pipeline dependency. ~4000 fps on CPU (OMP Release).
- **`JointStore::set_targets(const float*, const float*)`**: new method to update PD motor
  `target_pos` / `target_vel` per-frame without re-uploading the full joint SoA.
- **sign convention in `solve_ang_constraint`**: the angular impulse directions for body A and B
  were inverted; fixed — each body now receives `dl * I⁻¹n` with the correct sign such that the
  applied angular correction reduces, not amplifies, the violation.
- **per-GS-iteration quaternion normalisation**: quaternions are normalised after every
  Gauss-Seidel iteration (not just once at the end) to prevent drift-induced NaN when the
  angular correction runs for many iterations.
- **`ContactStore::reset()` required before first solve**: articulation tests that omit reset
  leave the contact count uninitialised, causing a SIGSEGV in the contact loop; all callers
  must reset before the first call to `solver.solve()`.
- **Articulation determinism gate** (`sim determinism: 2-link articulated arm, 20 frames`):
  static base + 2 revolute-jointed links, PD motors, gravity; 20 frames at 60 Hz; hash checked
  against a golden value in CI (`[determinism]` label, CPU/OMP Release golden: `0xa417e4fb155bea78`).

**Exit criterion (v0.2)**: ✅ `examples/manipulator_pick` runs headless, joints within limits,
block moves from home to place position, 4027 fps on CPU (~67x realtime). Articulation
determinism gate passes (2-link arm, 20 frames, golden hash hardcoded in CI).

### Phase 3 — Scene & Asset Pipeline
**Goal**: load complete indoor scenes.

- libsdformat integration for SDF/URDF loading.
- V-HACD / CoACD convex decomposition pipeline for non-convex meshes (CPU at load time, GPU buffers produced).
- Triangle-mesh BVH for static geometry; dynamic-vs-static narrowphase via mesh-BVH traversal on GPU.

**Exit criterion**: furnished room from an SDF world, mobile manipulator navigating and pushing objects, realtime.

### Phase 4 — Performance, Multi-GPU, Realtime Guarantees
**Goal**: lift the perf lid; explore multi-GPU.

- Spatial decomposition with halo-region exchange for single-scene multi-GPU.
- Pipelined async compute (overlap broadphase of frame N+1 with solver of frame N).
- Mixed precision for narrowphase candidate filtering only (never solver state).
- Realtime budget enforcer: timestep adjusts within bounds; never overruns wallclock budget.

**Exit criterion**: Phase 3 scene at ≥240 Hz single GPU, or ≥120 Hz scaled across two.

### Phase 5 — Sensor Hooks & API Polish
**Goal**: prepare for sensors without building them yet.

- Batched ray-query API on GPU (foundation for LiDAR, depth cameras, raycast sensors). Built on the same BVH used for broadphase.
- Contact-sensor API: per-shape contact event subscription, accessible from Python with zero-copy NumPy views (DLPack / CUDA array interface).
- Stable C++ ABI policy; versioned API.
- Documentation: tutorials, Doxygen + Sphinx, one worked example per robot class.

### Phase 6 (deferred) — Differentiability
Architecture in Phases 0–2 keeps this open. Add when usage justifies it: tape-based reverse-mode AD over the kernel framework, adjoint solver formulation, gradient checking against finite differences.

## Initial Repo Skeleton (files to be created in Phase 0)

```
.
├── CMakeLists.txt
├── CMakePresets.json                  # Ninja Multi-Config; configure × build ×
│                                      #   test × workflow presets (see Phase 0)
├── vcpkg.json                         # manifest: eigen3, spdlog, fmt, pugixml,
│                                      #   catch2, nanobind, tracy
├── vcpkg-configuration.json           # baseline pin
├── LICENSE                            # Apache-2.0
├── README.md
├── cmake/
│   ├── BackendOptions.cmake
│   ├── DependencyAllowlist.cmake      # device-safe header lint
│   └── DeterminismCI.cmake
├── compute/
│   ├── include/compute/
│   │   ├── device.hpp                 # Device, capability query
│   │   ├── buffer.hpp                 # Buffer<T>, upload/download
│   │   ├── kernel.hpp                 # parallel_for, ranges
│   │   ├── reduction.hpp              # deterministic reduce
│   │   ├── sort.hpp                   # sort_by_key
│   │   ├── atomics.hpp                # det-mode atomic ops
│   │   └── stream.hpp                 # Stream, Event
│   ├── src/sycl/                      # AdaptiveCpp impl
│   └── tests/                         # saxpy, reduce, sort, determinism
├── core/
│   ├── include/core/
│   │   ├── math/                      # device-safe Vec3, Quat, Mat3,
│   │   │                              #   Transform, SpatialVector, Inertia
│   │   ├── body.hpp                   # SoA body arrays
│   │   ├── integrator.hpp
│   │   ├── broadphase.hpp             # LBVH interface
│   │   ├── narrowphase.hpp            # primitive + GJK/EPA
│   │   ├── contact.hpp                # manifold types
│   │   ├── solver.hpp                 # IConstraintSolver interface
│   │   └── world.hpp                  # top-level simulation step
│   ├── src/
│   └── tests/                         # includes host-vs-device math equivalence
├── scene/
│   ├── include/scene/                 # later: SDF loader, scene graph
│   ├── src/
│   └── tests/
├── bindings/python/                   # nanobind module
├── examples/
│   ├── stress_test_blocks/            # v0.1 demo target — headless executable;
│   │                                  #   emits trajectory dump + metrics JSON
│   │                                  #   + determinism hash
│   └── (later) manipulator_pick/      # v0.2 demo target
├── tools/
│   └── viz/                           # standalone replay viewer (Python +
│                                      #   rerun.io / meshcat); manually invoked
│                                      #   on a workstation; not on the CI path
└── docs/
```

Notes on vcpkg usage:
- All deps resolved through `vcpkg.json`; no `FetchContent`, no `ExternalProject`, no vendored copies in `third_party/`.
- The vcpkg baseline SHA is pinned in `vcpkg-configuration.json` and bumped deliberately.
- AdaptiveCpp itself is *not* via vcpkg — it requires a system install (LLVM, CUDA/ROCm SDKs). Document the supported install paths in `README.md`.
- libsdformat (Phase 3): added to `vcpkg.json` at that time; transitive deps (ignition-math, ignition-utils, console_bridge, tinyxml2, protobuf) are handled by vcpkg.

## Verification

Each phase has a binary go/no-go test. CI must pass all gates from prior phases before a phase is considered shipped.

**Phase 0 gates**
- `ctest` on `compute/tests` passes on Linux + CUDA, Linux + AMD (HIP), Linux + CPU (OpenMP).
- Golden-trace regression: same input → bit-identical output across runs on same HW + build.
- Determinism harness used by at least one test (e.g. reduction over 1M random floats produces identical bytes across 10 runs).

**Phase 1 / v0.1 gates**
- `examples/stress_test_blocks` runs 1k+ bodies at ≥60 Hz on the RTX 3060 reference GPU, headless, asserted via metrics JSON in CI.
- Same demo runs on CPU fallback, slower but stable; same assertion mechanism, different threshold.
- Determinism harness extended to cover one full simulation step over a fixed scene; hash checked in CI.
- Benchmark suite (Catch2 `BENCHMARK(...)`) tracks per-frame timings broken down by broadphase / narrowphase / solver / integrator; results emitted as machine-readable JSON, tracked over time.

**Phase 2 / v0.2 gates**
- `examples/manipulator_pick` (hardcoded 7-DOF Franka-like arm) executes a scripted PD trajectory headless, picks a block, drops it elsewhere, ≥ 60 FPS on the reference CPU.
- Joint limits, PD motors, and contacts all exercised in the same simulation step.
- Determinism gate covers a 20-frame articulation trace (hash checked in CI).
- URDF loading is Phase 3; the Phase 2 gate uses the hardcoded robot.

**Phase 3+ gates**
- Scene-load round-trip: SDF in → simulation state → comparable to Gazebo's load of the same world for a defined comparison metric.
- Multi-GPU gate (Phase 4): same scene single-GPU vs two-GPU produces results within floating-point tolerance (cross-device determinism is not required).

**End-to-end smoke test (every phase) — fully headless, scripted, exits non-zero on any failure**
- Build clean from scratch (`cmake --workflow --preset=ci-linux-cuda` or equivalent).
- Run example demo headless for 60 seconds; assert metrics JSON satisfies budget thresholds.
- Run determinism harness; assert hash matches golden.
- Run Python-binding smoke test (`pytest bindings/python/tests`).
- Optional, never required for CI: human runs `tools/viz/replay.py <trajectory-dump>` on their workstation to eyeball the result.

## Risks

1. **GPU determinism cost** — typically 10–30% perf vs unconstrained atomics. Mitigation: deterministic-reduction primitives in the compute shim from day one, benchmarked.
2. **AdaptiveCpp pace / breakage** — pin a known-good version in CI; track upstream but don't auto-upgrade.
3. **Solo bandwidth on the kernel framework** — keep the compute shim minimal; resist adding features that aren't needed by the physics core.
4. **Multi-GPU single-scene is research-grade** — Phase 4 stretch, not a commitment. If hard, ship single-GPU v0.3 and revisit.
5. **Realtime on GPU is soft, not hard** — be honest in the API: "soft realtime, budget-enforced," not "hard realtime."
6. **Windows late-add may not be feasible** — accept this; do not constrain Phase 0–3 designs to keep Windows viable.

## Out of Scope (explicit)

- Rendering of any kind (RGB, photoreal, etc.)
- Sensor noise models, sensor data generation (planned Phase 5 hooks are geometry queries only)
- Soft bodies, cloth, deformables, fluids, particles
- Heightfield collision
- Signed-distance-field volume collision
- Mobile / embedded GPU targets
- Hard-realtime guarantees
- Differentiable physics (deferred, not precluded)
- Vulkan compute backend (deferred, not precluded)
- Windows first-class support (deferred)
