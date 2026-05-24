add_executable(test_compute_smoke smoke_test.cpp)
target_link_libraries(test_compute_smoke PRIVATE
    dyphur::compute
    Catch2::Catch2WithMain
)
dyphur_add_smoke_test(test_compute_smoke)
