add_executable(sdf_scene main.cpp)
target_link_libraries(sdf_scene PRIVATE dyphur::scene dyphur::core)
add_sycl_to_target(TARGET sdf_scene SOURCES main.cpp)

# Install the bundled world file next to the binary
add_custom_command(TARGET sdf_scene POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/box_room.sdf
        $<TARGET_FILE_DIR:sdf_scene>/box_room.sdf
    COMMENT "Copying box_room.sdf next to sdf_scene binary"
)
