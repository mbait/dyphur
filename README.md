# dyphur

GPU-first rigid-body physics framework for robotics simulation.

**Status**: Phase 1 complete. 1 024 rigid boxes settle on a static ground plane in realtime; output is deterministic (bit-identical hash) across runs on the same hardware and build. See [docs/PLAN.md](docs/PLAN.md) for the full design and roadmap.

## Goals

- **GPU-first**, vendor-neutral compute via AdaptiveCpp/SYCL (CUDA, HIP, OpenMP-CPU backends).
- **Realtime** simulation of indoor scenes containing robots whose construction is expressible in SDF/URDF (manipulators, drones, mobile robots).
- **Deterministic** on a given build + hardware combination, gated in CI.
- **Headless-first**: every check runs without a display. Visualization is a separate, manually-invoked replay tool.
- **C++20 core** with Python bindings via nanobind.
- **Apache-2.0** licensed.

## Non-goals (explicit)

Rendering, sensor noise simulation, soft bodies, fluids, mobile/embedded GPU targets, hard-realtime guarantees. See [docs/PLAN.md § Out of Scope](docs/PLAN.md#out-of-scope-explicit).

## Technical overview

### Compute layer (`compute/`)

A thin, header-only C++ shim over AdaptiveCpp/SYCL 2020. Provides:

- `Device` / `Stream` — queue ownership and lifetime
- `Buffer<T>` — USM device memory with typed upload/download
- `parallel_for` — work-item dispatch, host and device backends
- `sort_by_key` — deterministic bitonic sort (power-of-2 range)
- `atomic_add_seq` — sequentially-consistent fetch-add for lock-free counters
- `reduce` — parallel tree reduction

No STL containers, exceptions, or virtual functions cross into device code. The `cmake/DependencyAllowlist.cmake` lint target enforces this boundary.

### Physics pipeline (`core/`)

One fixed-size frame step: **integrate → broadphase → sort pairs → narrowphase → solve**.

**Data layout**: SoA (structure-of-arrays) throughout all hot paths. Body state (`BodyView`) stores position, linear/angular velocity, quaternion orientation, inverse mass, and body-frame inverse inertia as separate flat float arrays — SIMD- and GPU-cache-friendly.

**Integrator**: Symplectic (semi-implicit) Euler. Velocity updated from forces first, then position from velocity. Substep count is a runtime parameter.

**Broadphase**: LBVH (Linear Bounding Volume Hierarchy, Karras 2012). Bodies are quantized to 10-bit Morton codes, sorted by Morton code (bitonic sort), and the binary radix tree is built in a single parallel kernel. AABBs are refitted bottom-up with per-node atomic flags (acquire-release). Traversal emits overlapping candidate pairs (leaf_i < leaf_j) using the BVH from each leaf as a query. Pairs are sorted by canonical `(a,b)` key after collection to ensure deterministic downstream ordering.

**Narrowphase**: Per-pair SAT-based contact detection, processed sequentially (single work-item) over the sorted pair list for determinism. Supports sphere–sphere, sphere–box, and box–box. Box–box uses 15 SAT axes (3 face normals × 2 + 9 edge–edge cross products) and generates up to 4 vertex-face contact points per pair via a face-area heuristic: the larger body's face is always chosen as the reference so that the smaller body's vertices (which lie within the larger face's footprint) are tested, avoiding the zero-contact failure mode that occurs when a small box sits on a large ground plane.

**Solver**: Extended Position-Based Dynamics (XPBD, Müller et al. 2020). Sequential Gauss-Seidel, single GPU work-item for determinism. Full rigid-body angular response: generalized inverse mass includes the angular term `(r×n)·(I_world⁻¹(r×n))` where `I_world⁻¹ = R·I_body⁻¹·Rᵀ`. Quaternion corrections applied each iteration, followed by re-normalization. Zero-restitution velocity correction zeroes the normal relative velocity at contact. Per-contact accumulated lambda prevents multi-iteration over-correction when a body pair has several contact points.

**Determinism**: Achieved by (1) Morton-code ordering of the LBVH, (2) bitonic sort of broadphase pairs by `(a,b)` key, (3) sequential narrowphase processing in sorted-pair order, and (4) single-work-item XPBD with fixed contact traversal order. Same build + same hardware → bit-identical final state hash across runs.

### Output format (`stress_test_blocks`)

The Phase 1 demo executable emits three files per run:

| File | Content |
|---|---|
| `<prefix>.trajectory` | Binary stream: 2×`uint32` header `(n_bodies, n_frames)`, then per-frame rows of 7×`float` `(x, y, z, qw, qx, qy, qz)` per body, sampled at 30 Hz |
| `<prefix>.metrics.json` | Wall-clock time, frames-per-second, realtime factor, average contacts/frame |
| `<prefix>.golden` | FNV-1a-64 hex hash of the final body state (position + orientation of all bodies) |

## Simulation results

Benchmark: `stress_test_blocks` — 1 024 dynamic boxes (0.4 m half-extent, 1 kg) stacked 8×8×16 above a static ground plane (20 m × 0.5 m × 20 m), simulated for 300 frames at 60 Hz (5 s simulated time). Release build, no GPU SDK required for the CPU backend.

| Processing Unit | Backend | Bodies | Avg contacts/frame | FPS | Realtime factor | Deterministic |
|---|---|---|---|---|---|---|
| Intel Xeon E5-2667 v4 @ 3.20 GHz | OpenMP (CPU) | 1 025 | 2 615 | 254 | 4.24× | yes (`7168a504ac96d97e`) |

## Requirements

- Linux (primary; Windows deferred)
- CMake ≥ 3.25
- Ninja
- A C++20 compiler (GCC 12+, Clang 15+)
- vcpkg (bundled as a git submodule; no `VCPKG_ROOT` needed)
- AdaptiveCpp (installed system-wide; not via vcpkg)
- For the CUDA backend: NVIDIA driver + CUDA Toolkit (development on RTX 3060, compute capability 8.6)
- For the HIP backend: ROCm
- CPU-only builds need none of the above GPU SDKs

## Quick start

```sh
# After cloning, initialize vcpkg
git submodule update --init

# Local development loop (Debug build, OMP/CPU backend, smoke tests)
cmake --workflow --preset=dev

# Run the Phase 1 stress test (CPU backend)
cmake --preset=omp
cmake --build build/omp --config Release --target stress_test_blocks
./build/omp/examples/stress_test_blocks/Release/stress_test_blocks

# CI-equivalent runs
cmake --workflow --preset=ci-linux-cuda
cmake --workflow --preset=ci-linux-hip
cmake --workflow --preset=ci-linux-cpu
cmake --workflow --preset=ci-determinism
```

## Repo layout

| Path | Purpose |
|---|---|
| `compute/` | Thin C++ shim over AdaptiveCpp/SYCL — `Device`, `Buffer`, `parallel_for`, deterministic reductions, sort, atomics |
| `core/` | Physics core: integrator, broadphase, narrowphase, XPBD solver, body/shape stores |
| `scene/` | SDF/URDF loading via libsdformat; scene graph; convex decomposition pipeline (Phase 3+) |
| `bindings/python/` | nanobind Python module (Phase 4+) |
| `examples/` | Headless executables; emit state-trajectory dumps + metrics JSON + determinism hashes |
| `tools/viz/` | Standalone replay viewer for trajectory dumps. Manually invoked; not on the CI path. |
| `cmake/` | Build helpers: backend selection, device-header allow-list lint, determinism harness |
| `docs/` | Plan, design notes |

## License

Apache-2.0. See [LICENSE](LICENSE).
