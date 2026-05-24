add_library(dyphur_core INTERFACE)
add_library(dyphur::core ALIAS dyphur_core)

target_include_directories(dyphur_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(dyphur_core INTERFACE
    dyphur::compute
    Eigen3::Eigen
)

# Physics core implementation files land here in Phases 1-2:
#   target_sources(dyphur_core PRIVATE src/body.cpp src/integrator.cpp ...)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
