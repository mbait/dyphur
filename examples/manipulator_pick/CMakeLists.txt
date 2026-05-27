add_executable(manipulator_pick main.cpp)
target_link_libraries(manipulator_pick PRIVATE dyphur::core)
add_sycl_to_target(TARGET manipulator_pick SOURCES main.cpp)
