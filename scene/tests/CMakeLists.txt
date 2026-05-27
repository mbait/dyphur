add_executable(test_scene_smoke smoke_test.cpp sdf_loader_test.cpp)
target_link_libraries(test_scene_smoke PRIVATE
    dyphur::scene
    Catch2::Catch2WithMain
)
dyphur_add_smoke_test(test_scene_smoke)
