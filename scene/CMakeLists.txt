add_library(dyphur_scene INTERFACE)
add_library(dyphur::scene ALIAS dyphur_scene)

target_include_directories(dyphur_scene INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(dyphur_scene INTERFACE
    dyphur::core
    pugixml::pugixml
)

# Phase 3 brings in libsdformat (gated via vcpkg "scene" feature) and the
# convex-decomposition pipeline. Until then this is an empty INTERFACE
# library that exists so the rest of the build can reference it.

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
