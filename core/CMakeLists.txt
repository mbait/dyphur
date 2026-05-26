add_library(dyphur_core INTERFACE)
add_library(dyphur::core ALIAS dyphur_core)

target_include_directories(dyphur_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(dyphur_core INTERFACE
    dyphur::compute
    Eigen3::Eigen
    spdlog::spdlog
    fmt::fmt
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
