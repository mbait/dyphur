# tools/viz — Trajectory Visualiser

Standalone executable for visualising dyphur trajectory dumps.  Not on the CI
path; built only when Magnum is found in vcpkg.

## Modes

```
viz replay   <prefix> [--fps N] [--loop]
viz snapshot <prefix> [--frame N] [-o out.png] [--width W] [--height H]
viz live     <prefix.scene>          (not yet implemented)
```

### replay — interactive trajectory playback

Opens a GLFW window.  Reads `<prefix>.trajectory` and `<prefix>.scene`.

Controls:
- Space — pause / unpause
- ← → arrow keys — step one frame (pauses automatically)
- Left-drag — orbit camera
- Scroll — zoom in / out
- Q / Escape — quit

### snapshot — headless PNG output

EGL offscreen rendering (no display required).  Renders one trajectory frame
and writes a PNG.  Useful for whitepaper figures.

```
viz snapshot stress_test_blocks --frame 0 -o frame0.png
```

Default output size: 1280×720.

### live — real-time visualisation (not yet implemented)

Will read poses from a POSIX shared-memory ring buffer written by a running
simulation via `dyphur::VizSink`.

## Build

vcpkg installs Magnum automatically when you run cmake.  Build with any preset:

```
cmake --preset=omp
cmake --build build/omp --target viz
```

If Magnum is unavailable, cmake prints a warning and skips `tools/viz`.
Pass `-DDYPHUR_SKIP_VIZ=ON` to suppress even that.

## Output files

Each demo writes `<prefix>.trajectory` and `<prefix>.scene` alongside the
existing `.metrics.json` and `.golden` outputs.  The `.scene` file maps each
body index to a shape (type + half-extents) so the visualiser can render the
correct geometry.
