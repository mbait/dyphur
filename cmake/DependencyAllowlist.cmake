# Device-safe header allow-list.
#
# Headers under any `kernels/` subdirectory (and any *.kernel.hpp file) may
# only #include from the list below. Everything else — STL containers that
# allocate, Eigen, spdlog, std::filesystem, etc. — is forbidden in kernel
# translation units, because they don't compile on GPU backends.
#
# The lint that enforces this is registered as a custom target. Run it via:
#   cmake --build <build-dir> --target dyphur-lint-device-headers
#
# Implementation of the lint script itself is deferred (Phase 0 deliverable);
# this file declares the allow-list and the target so the rest of the build
# can reference them stably.

set(DYPHUR_DEVICE_SAFE_HEADERS
    # C++ standard headers that are constexpr-friendly, allocation-free,
    # and exception-free.
    "<cstdint>"
    "<cstddef>"
    "<cmath>"
    "<array>"
    "<type_traits>"
    "<utility>"
    "<limits>"
    "<bit>"
    # Our own device-safe math primitives.
    "core/math/vec3.hpp"
    "core/math/quat.hpp"
    "core/math/mat3.hpp"
    "core/math/transform.hpp"
    "core/math/spatial.hpp"
    "core/math/inertia.hpp"
    # Compute primitives that wrap SYCL types without leaking them.
    "compute/device.hpp"
    "compute/stream.hpp"
    "compute/buffer.hpp"
    "compute/kernel.hpp"
    "compute/reduction.hpp"
    "compute/sort.hpp"
    "compute/atomics.hpp"
    "compute/compute.hpp"
    # Convenience umbrella for all math types.
    "core/math/math.hpp"
)

add_custom_target(dyphur-lint-device-headers
    COMMAND ${CMAKE_COMMAND} -E echo
        "[dyphur] device-header allow-list lint: not yet implemented (see cmake/DependencyAllowlist.cmake)"
    VERBATIM
)
