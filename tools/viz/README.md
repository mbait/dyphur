# tools/viz — Standalone Trajectory Viewer

This is the **only** part of the project intended for manual human use. It is **not** part of the engine, **not** on the CI path, and **not** required for any automated check.

## Purpose

Read a `.trajectory` dump produced by a headless dyphur example (e.g. `examples/stress_test_blocks`) and replay it visually so a human can sanity-check what happened.

## Status

Not yet implemented. Planned as a Python script using one of:

- [`rerun.io`](https://rerun.io) — Apache-2, modern, robotics-friendly, separate viewer GUI
- [`meshcat`](https://github.com/meshcat-dev/meshcat) — minimal browser-based viewer, common in robotics

## Why it's separate

The engine itself is headless-first. All correctness and performance claims are backed by automated tests / benchmarks / golden-hash regressions. Visualization is for human comprehension only, never for validation. Keeping the viewer out of the build avoids dragging GUI / display dependencies onto headless CI machines.
