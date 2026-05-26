add_library(dyphur_core STATIC
    src/body_store.cpp
    src/integrator.cpp
    src/shapes.cpp
    src/broadphase.cpp
    src/contact_store.cpp
    src/narrowphase.cpp
    src/xpbd_solver.cpp
)
add_library(dyphur::core ALIAS dyphur_core)
add_sycl_to_target(TARGET dyphur_core SOURCES
    src/body_store.cpp
    src/integrator.cpp
    src/shapes.cpp
    src/broadphase.cpp
    src/contact_store.cpp
    src/narrowphase.cpp
    src/xpbd_solver.cpp
)

target_include_directories(dyphur_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(dyphur_core PUBLIC
    dyphur::compute
    Eigen3::Eigen
    spdlog::spdlog
    fmt::fmt
)

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
