add_executable(test_compute_smoke smoke_test.cpp)
target_link_libraries(test_compute_smoke PRIVATE
    dyphur::compute
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_compute_smoke SOURCES smoke_test.cpp)
dyphur_add_smoke_test(test_compute_smoke)

add_executable(test_compute_determinism determinism_test.cpp)
target_link_libraries(test_compute_determinism PRIVATE
    dyphur::compute
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_compute_determinism SOURCES determinism_test.cpp)
dyphur_add_determinism_test(test_compute_determinism)
