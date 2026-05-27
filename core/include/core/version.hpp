#pragma once

// Project version constants, generated from CMakeLists.txt project() declaration.
#define DYPHUR_VERSION_MAJOR 0
#define DYPHUR_VERSION_MINOR 0
#define DYPHUR_VERSION_PATCH 1

#define DYPHUR_VERSION_STRING "0.0.1"

// Encode as a single integer for numeric comparisons.
#define DYPHUR_VERSION \
    ((DYPHUR_VERSION_MAJOR) * 10000 + \
     (DYPHUR_VERSION_MINOR) *   100 + \
     (DYPHUR_VERSION_PATCH))
