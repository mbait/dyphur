add_executable(stress_test_blocks main.cpp)
target_link_libraries(stress_test_blocks PRIVATE dyphur::core)
add_sycl_to_target(TARGET stress_test_blocks SOURCES main.cpp)
