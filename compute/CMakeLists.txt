add_library(dyphur_compute INTERFACE)
add_library(dyphur::compute ALIAS dyphur_compute)

target_include_directories(dyphur_compute INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# Consumers that include compute headers contain SYCL kernel lambdas and must
# be compiled with AdaptiveCpp. Call add_sycl_to_target() on each such target.

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
