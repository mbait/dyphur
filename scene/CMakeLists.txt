find_package(sdformat15 CONFIG QUIET)
if(NOT sdformat15_FOUND)
    # Try unversioned fallback (some distributions install it without the version suffix)
    find_package(sdformat CONFIG QUIET)
    if(sdformat_FOUND)
        set(sdformat15_FOUND TRUE)
        if(NOT TARGET sdformat15::sdformat15)
            add_library(sdformat15::sdformat15 ALIAS sdformat::sdformat)
        endif()
    endif()
endif()

if(NOT sdformat15_FOUND)
    message(WARNING
        "libsdformat not found — scene module will be built as an empty stub.\n"
        "Install sdformat via vcpkg ('sdformat' port) or system package manager.")
endif()

add_library(dyphur_scene STATIC)
add_library(dyphur::scene ALIAS dyphur_scene)

target_include_directories(dyphur_scene PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

target_link_libraries(dyphur_scene PUBLIC
    dyphur::core
    pugixml::pugixml
)

if(sdformat15_FOUND)
    find_path(V_HACD_INCLUDE_DIRS "VHACD.h")
    target_sources(dyphur_scene PRIVATE
        src/sdf_loader.cpp
        src/convex_decomp.cpp
    )
    target_link_libraries(dyphur_scene PRIVATE sdformat15::sdformat15)
    if(V_HACD_INCLUDE_DIRS)
        target_include_directories(dyphur_scene PRIVATE ${V_HACD_INCLUDE_DIRS})
    endif()
    target_compile_definitions(dyphur_scene PUBLIC DYPHUR_SCENE_ENABLED)
else()
    # Empty stubs so targets that link dyphur::scene still compile
    target_sources(dyphur_scene PRIVATE src/scene_stub.cpp)
endif()

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
