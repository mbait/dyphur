# Discover nanobind from the project virtualenv (.venv).
# The venv path is relative to the source root; make it absolute.
set(NB_VENV "${CMAKE_SOURCE_DIR}/.venv")
execute_process(
    COMMAND "${NB_VENV}/bin/python3" -c "import nanobind; print(nanobind.cmake_dir())"
    OUTPUT_VARIABLE NB_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE NB_QUERY_RESULT
)
if(NOT NB_QUERY_RESULT EQUAL 0)
    message(FATAL_ERROR
        "Could not locate nanobind via ${NB_VENV}/bin/python3. "
        "Run: python3 -m venv .venv && .venv/bin/pip install nanobind")
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind REQUIRED CONFIG PATHS "${NB_CMAKE_DIR}" NO_DEFAULT_PATH)

nanobind_add_module(dyphur_py
    NB_STATIC
    dyphur_py.cpp
)

target_link_libraries(dyphur_py PRIVATE dyphur::core)

# dyphur headers include <sycl/sycl.hpp> transitively; compile with AdaptiveCpp.
# No device kernels in this file — the flag only enables the header include path.
add_sycl_to_target(TARGET dyphur_py SOURCES dyphur_py.cpp)

# Install the .so next to the Python tests so they can import it directly.
set_target_properties(dyphur_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/tests"
)

if(BUILD_TESTING)
    # Symlink the built .so into tests/ so pytest can import dyphur_py without
    # having to manipulate PYTHONPATH.  The symlink is regenerated each build.
    add_custom_command(TARGET dyphur_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            "$<TARGET_FILE:dyphur_py>"
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/$<TARGET_FILE_NAME:dyphur_py>"
        COMMENT "Symlinking dyphur_py.so into tests/"
    )

    add_test(NAME pytest_contact_sensor
        COMMAND "${NB_VENV}/bin/python3" -m pytest tests/ -v
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
    set_tests_properties(pytest_contact_sensor PROPERTIES LABELS "smoke")
endif()
