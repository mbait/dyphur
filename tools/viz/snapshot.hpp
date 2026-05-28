#pragma once
#include <string>

namespace dyphur::viz {

// Run snapshot mode.  Returns 0 on success.
// argv[0] is the program name; remaining argv is ["snapshot", prefix,
// "--frame", N, "-o", outfile, [--width W] [--height H]].
int run_snapshot(int argc, char** argv);

} // namespace dyphur::viz
