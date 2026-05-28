if(DYPHUR_SKIP_VIZ)
    return()
endif()

# Magnum's FindMagnum.cmake calls find_package(EGL) but CMake 3.28 has no
# built-in FindEGL.cmake. Provide ours before Magnum searches for components.
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
find_package(EGL REQUIRED)

find_package(Magnum QUIET COMPONENTS
    GL MeshTools Primitives Shaders Trade
    GlfwApplication WindowlessEglApplication)
find_package(MagnumPlugins QUIET COMPONENTS StbImageConverter)

if(NOT Magnum_FOUND OR NOT MagnumPlugins_FOUND)
    message(STATUS
        "tools/viz: Magnum or MagnumPlugins not found — skipping. "
        "Pass -DDYPHUR_SKIP_VIZ=ON to suppress this message.")
    return()
endif()

add_executable(viz
    main.cpp
    renderer.cpp
    snapshot.cpp
    replay.cpp
    live.cpp
)

target_link_libraries(viz PRIVATE
    dyphur::core
    Magnum::GL
    Magnum::MeshTools
    Magnum::Primitives
    Magnum::Shaders
    Magnum::Trade
    Magnum::GlfwApplication
    Magnum::WindowlessEglApplication
    MagnumPlugins::StbImageConverter
)

target_compile_features(viz PRIVATE cxx_std_20)

# examples/common is in the source tree alongside examples; expose it explicitly.
target_include_directories(viz PRIVATE
    ${PROJECT_SOURCE_DIR}/examples)
