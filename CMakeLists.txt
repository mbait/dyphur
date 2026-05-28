cmake_minimum_required(VERSION 3.25)

project(dyphur
    VERSION 0.0.1
    LANGUAGES CXX
    DESCRIPTION "GPU-first rigid-body physics framework for robotics simulation"
    HOMEPAGE_URL "https://github.com/mbait/dyphur"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)  # required for Python shared module

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(BackendOptions)
include(DeterminismCI)
include(DependencyAllowlist)

find_package(Eigen3   CONFIG REQUIRED)
find_package(spdlog   CONFIG REQUIRED)
find_package(fmt      CONFIG REQUIRED)
find_package(pugixml  CONFIG REQUIRED)
find_package(Catch2 3 CONFIG REQUIRED)

include(CTest)
enable_testing()
include(Catch)

add_subdirectory(compute)
add_subdirectory(core)
add_subdirectory(scene)
add_subdirectory(examples)
add_subdirectory(bindings/python)

add_subdirectory(tools/viz)

# ── Install / package export ──────────────────────────────────────────────────
include(CMakePackageConfigHelpers)

# ── Documentation ─────────────────────────────────────────────────────────────
find_program(DOXYGEN_EXECUTABLE doxygen)
set(NB_VENV_SPHINX "${CMAKE_SOURCE_DIR}/.venv/bin/sphinx-build")

if(DOXYGEN_EXECUTABLE AND EXISTS "${NB_VENV_SPHINX}")
    add_custom_target(docs
        COMMAND "${DOXYGEN_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/docs/Doxyfile"
        COMMAND "${NB_VENV_SPHINX}" -b html
                "${CMAKE_SOURCE_DIR}/docs"
                "${CMAKE_SOURCE_DIR}/docs/_build/html"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/docs"
        COMMENT "Generating Doxygen XML then Sphinx HTML documentation"
        VERBATIM
    )
endif()

install(TARGETS dyphur_compute
        EXPORT dyphurTargets)

install(EXPORT dyphurTargets
        FILE dyphurTargets.cmake
        NAMESPACE dyphur::
        DESTINATION lib/cmake/dyphur)

configure_package_config_file(
    cmake/dyphurConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/dyphurConfig.cmake"
    INSTALL_DESTINATION lib/cmake/dyphur
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dyphurConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dyphurConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dyphurConfigVersion.cmake"
    DESTINATION lib/cmake/dyphur
)
