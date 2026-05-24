// v0.1 demo target: headless rigid-body stress test.
//
// When implemented, this executable simulates a pile/domino scene with
// 1k+ rigid bodies for a fixed wall-clock budget, then writes:
//   - <out>.trajectory      binary per-frame state dump for tools/viz/
//   - <out>.metrics.json    steps/sec, contacts/frame, peak VRAM, etc.
//   - <out>.golden          determinism hash of the final state
//
// No window, no graphics context. Visualization is a separate manual step.

#include <cstdio>

int main(int /*argc*/, char** /*argv*/) {
    std::puts("dyphur stress_test_blocks: not yet implemented (Phase 1 deliverable)");
    return 0;
}
