# dyphur

GPU-first rigid-body physics framework for robotics simulation.

**Status**: Phase 0 (foundations). Nothing simulates yet. See [docs/PLAN.md](docs/PLAN.md) for the full design and roadmap.

## Goals

- **GPU-first**, vendor-neutral compute via AdaptiveCpp/SYCL (CUDA, HIP, OpenMP-CPU backends).
- **Realtime** simulation of indoor scenes containing robots whose construction is expressible in SDF/URDF (manipulators, drones, mobile robots).
- **Deterministic** on a given build + hardware combination, gated in CI.
- **Headless-first**: every check runs without a display. Visualization is a separate, manually-invoked replay tool.
- **C++20 core** with Python bindings via nanobind.
- **Apache-2.0** licensed.

## Non-goals (explicit)

Rendering, sensor noise simulation, soft bodies, fluids, mobile/embedded GPU targets, hard-realtime guarantees. See [docs/PLAN.md § Out of Scope](docs/PLAN.md#out-of-scope-explicit).

## Requirements

- Linux (primary; Windows deferred)
- CMake ≥ 3.25
- Ninja
- A C++20 compiler (GCC 12+, Clang 15+)
- vcpkg (installed; `VCPKG_ROOT` set)
- AdaptiveCpp (installed system-wide; not via vcpkg)
- For the CUDA backend: NVIDIA driver + CUDA Toolkit (development on RTX 3060, compute capability 8.6)
- For the HIP backend: ROCm
- CPU-only builds need none of the above GPU SDKs

## Quick start (once Phase 0 is implemented)

```sh
# Local development loop (Debug build, CUDA backend, smoke tests)
cmake --workflow --preset=dev

# CI-equivalent run (Release build, CUDA backend, full test suite)
cmake --workflow --preset=ci-linux-cuda
```

## Repo layout

| Path | Purpose |
|---|---|
| `compute/` | Thin C++ shim over AdaptiveCpp/SYCL — `Device`, `Buffer`, `parallel_for`, deterministic reductions, sort, atomics |
| `core/` | Physics core: bodies, articulations, broadphase, narrowphase, contact, solver interface, integrator, world |
| `scene/` | SDF/URDF loading via libsdformat; scene graph; convex decomposition pipeline (Phase 3) |
| `bindings/python/` | nanobind Python module |
| `examples/` | Headless example executables that emit state-trajectory dumps + metrics JSON + determinism hashes |
| `tools/viz/` | Standalone replay viewer for state-trajectory dumps (Python). Manually invoked. Not on the CI path. |
| `cmake/` | Build helpers: backend selection, device-header allow-list lint, determinism harness |
| `docs/` | Plan, design notes |

## License

Apache-2.0. See [LICENSE](LICENSE).
