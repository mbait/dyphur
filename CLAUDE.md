# dyphur — Project Context for Claude

This file is the canonical entry point for any Claude session resuming work on this project. Read it first.

## What this is

`dyphur` is a GPU-first rigid-body physics framework for robotics simulation. The full design rationale, phased roadmap, and verification gates live in [docs/PLAN.md](docs/PLAN.md). **Always read `docs/PLAN.md` before doing nontrivial work.**

Project owner: Alexander Solovets (asolovets@gmail.com). Solo, full-time.

## Locked decisions (do not relitigate without explicit user request)

These were debated extensively in the planning phase and are settled:

- **Language**: C++20 core + Python bindings via nanobind.
- **Compute**: AdaptiveCpp (SYCL 2020). Backends: CUDA, HIP, OpenMP-CPU. Level Zero optional. *Vulkan compute is deferred indefinitely — do not propose it.*
- **Build system**: CMake ≥ 3.25 with **Ninja Multi-Config**, presets, and workflows from day 1.
- **Package manager**: vcpkg (manifest mode). All deps except AdaptiveCpp.
- **License**: Apache-2.0.
- **Platform**: Linux primary. Windows deferred to v0.3+ or later. No mobile, ever.
- **Reference GPU**: NVIDIA RTX 3060 (Ampere, compute capability 8.6, 12 GB VRAM).
- **Solver**: XPBD as the first/reference implementation behind a solver-agnostic interface.
- **Determinism**: Bit-identical on same hardware + same build. CI gate on every PR.
- **Data layout**: SoA / AoSoA everywhere in hot paths. Persistent across frames (keeps differentiability open).
- **Math (host)**: Eigen. Mandatory for robotics interop.
- **Math (device)**: Hand-rolled SoA-friendly types (`Vec3`, `Quat`, `Mat3`, `Transform`, `SpatialVector`, `Inertia`). Eigen is **explicitly excluded** from device code — its SYCL story is unofficial.
- **Scene format**: SDF/URDF via libsdformat (handles URDF→SDF internally).
- **Collision shapes (v0.1–v0.2)**: primitives + convex hulls (V-HACD) + triangle mesh. No heightfield, no SDF-volume.
- **Tests + benchmarks**: Catch2 v3 (includes bundled microbenchmarks; no separate benchmark dep).

## The non-negotiable rules

1. **SoA / AoSoA everywhere in hot paths.** No AoS data crosses into a kernel.
2. **Solver is an interface, not an implementation.** The core ships with XPBD; alternates must be addable without touching the rest of the physics core.
3. **Determinism is a CI gate, not a hope.** Every PR runs a golden-trace regression.
4. **Headless-first.** All tests, benchmarks, determinism checks, and demos run without a display. The single exception is the standalone visualization tool in `tools/viz/`, which a human invokes manually against state-trajectory dumps. No "build it and look at it" verification.
5. **Dependency boundary.** Third-party C++ libraries fall into four roles (see `docs/PLAN.md § Dependency Policy`). Kernel code only sees the device-safe allow-list — CI-enforced.
6. **The only physics-related thing we may not reuse is entire physics engines or their solvers.** Everything else (math, geometry, collision-detection algorithms like GJK/EPA, BVH builders, asset loaders) is fair game.

## Current state

- **Phase 0 (Foundations)**: skeleton only. Build system stubs, empty include dirs, smoke tests that pass trivially. No real implementation yet.
- v0.1 target = end of Phase 1 (rigid-body stress test, 1k+ bodies realtime, headless executable emitting trajectory dump + metrics JSON + determinism hash).

## How to work in this repo

- Local dev loop: `cmake --workflow --preset=dev` (once Phase 0 is implemented).
- CI-equivalent run: `cmake --workflow --preset=ci-linux-cuda` (or `-hip`, `-cpu`, `-determinism`).
- Build directories are `build/<preset-name>/`.
- AdaptiveCpp is **not** a vcpkg dep — it requires a system install (LLVM + CUDA/ROCm SDKs). User installs it per platform docs.

## What to do when stuck

- For design questions: re-read `docs/PLAN.md`. If still ambiguous, ask the user — do not invent.
- For build issues: check `cmake/BackendOptions.cmake`, `CMakePresets.json`, and `vcpkg.json` together. They are the source of truth for build configuration.
- For "should I add this dep / feature / abstraction" questions: default to no. Match the existing minimalism. The user prefers terse, dependency-light code.

## Communication style preferences

- Terse responses. No trailing "I've done X, Y, Z" summaries when the diff already shows it.
- Honest uncertainty over false confidence. Flag toolchain risks (especially AdaptiveCpp Windows or AMD edges) rather than hand-waving them.
- One sentence per status update; complete sentences; reader can pick up cold.
