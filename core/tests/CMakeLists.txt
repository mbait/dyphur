# Smoke test: pure host-side math type tests, no SYCL required.
add_executable(test_core_smoke smoke_test.cpp)
target_link_libraries(test_core_smoke PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
dyphur_add_smoke_test(test_core_smoke)

# Equivalence test: runs math ops on host and device, asserts bit-identical.
add_executable(test_core_math_equiv math_equiv_test.cpp)
target_link_libraries(test_core_math_equiv PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_math_equiv SOURCES math_equiv_test.cpp)
dyphur_add_smoke_test(test_core_math_equiv)

# BodyStore test: upload/download round-trip, requires SYCL.
add_executable(test_core_body body_store_test.cpp)
target_link_libraries(test_core_body PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_body SOURCES body_store_test.cpp)
dyphur_add_smoke_test(test_core_body)

# Integrator test: symplectic Euler correctness, requires SYCL.
add_executable(test_core_integrator integrator_test.cpp)
target_link_libraries(test_core_integrator PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_integrator SOURCES integrator_test.cpp)
dyphur_add_smoke_test(test_core_integrator)

# Broadphase test: LBVH build + pair detection, requires SYCL.
add_executable(test_core_broadphase broadphase_test.cpp)
target_link_libraries(test_core_broadphase PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_broadphase SOURCES broadphase_test.cpp)
dyphur_add_smoke_test(test_core_broadphase)
