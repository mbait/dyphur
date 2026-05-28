#pragma once

namespace dyphur::viz {

// Run replay mode.  Returns 0 when the window is closed.
// argv slice starts at program name; remaining: ["replay", prefix, [--fps N], [--loop]].
int run_replay(int argc, char** argv);

} // namespace dyphur::viz
