// viz — dyphur trajectory visualiser
//
// Modes:
//   viz replay  <prefix> [--fps N] [--loop]
//   viz snapshot <prefix> [--frame N] [-o out.png] [--width W] [--height H]
//   viz live    <prefix.scene> [--port P]   (not yet implemented)

#include "replay.hpp"
#include "snapshot.hpp"
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
            "usage: viz <mode> ...\n"
            "  viz replay   <prefix> [--fps N] [--loop]\n"
            "  viz snapshot <prefix> [--frame N] [-o out.png] [--width W] [--height H]\n"
            "  viz live     <prefix.scene>  (not yet implemented)\n");
        return 1;
    }

    const char* mode = argv[1];
    // Pass the full argv to the mode handler so it can forward to Magnum.
    if (std::strcmp(mode, "replay") == 0)
        return dyphur::viz::run_replay(argc, argv);
    if (std::strcmp(mode, "snapshot") == 0)
        return dyphur::viz::run_snapshot(argc, argv);
    if (std::strcmp(mode, "live") == 0) {
        std::fprintf(stderr, "viz: live mode not yet implemented\n");
        return 1;
    }

    std::fprintf(stderr, "viz: unknown mode '%s'\n", mode);
    return 1;
}
