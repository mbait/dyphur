add_executable(test_core_smoke smoke_test.cpp)
target_link_libraries(test_core_smoke PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
dyphur_add_smoke_test(test_core_smoke)
