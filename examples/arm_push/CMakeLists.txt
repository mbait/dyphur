add_executable(arm_push main.cpp)
add_sycl_to_target(TARGET arm_push)

target_link_libraries(arm_push PRIVATE
    dyphur::scene
    dyphur::core
)

add_custom_command(TARGET arm_push POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${CMAKE_CURRENT_SOURCE_DIR}/arm_room.sdf
        $<TARGET_FILE_DIR:arm_push>/arm_room.sdf
)

if(BUILD_TESTING)
    add_test(NAME test_arm_push_smoke
        COMMAND arm_push ${CMAKE_CURRENT_SOURCE_DIR}/arm_room.sdf 60
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    )
    set_tests_properties(test_arm_push_smoke PROPERTIES LABELS "smoke")
endif()
