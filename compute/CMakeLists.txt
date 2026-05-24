add_library(dyphur_compute INTERFACE)
add_library(dyphur::compute ALIAS dyphur_compute)

target_include_directories(dyphur_compute INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(dyphur_compute INTERFACE
    Eigen3::Eigen
    spdlog::spdlog
    fmt::fmt
)

# Backend-specific sources land here in Phase 0:
#   if(DYPHUR_BACKEND STREQUAL "CUDA") target_sources(... src/sycl/...) endif()
# For now this is a header-only INTERFACE target so the rest of the build
# works against a stable surface.

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
