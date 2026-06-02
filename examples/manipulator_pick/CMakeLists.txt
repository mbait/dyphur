add_executable(manipulator_pick main.cpp)
target_link_libraries(manipulator_pick PRIVATE dyphur::core dyphur::scene)
add_sycl_to_target(TARGET manipulator_pick SOURCES main.cpp)

# Copy Panda assets next to the executable so runtime path resolution works.
add_custom_command(TARGET manipulator_pick POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        $<TARGET_FILE_DIR:manipulator_pick>/assets
    COMMENT "Copying Panda assets"
)
