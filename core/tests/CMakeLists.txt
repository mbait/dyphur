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

# Articulation test: JointStore upload/download round-trip, requires SYCL.
add_executable(test_core_articulation articulation_test.cpp)
target_link_libraries(test_core_articulation PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_articulation SOURCES articulation_test.cpp)
dyphur_add_smoke_test(test_core_articulation)

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

# Narrowphase test: sphere-sphere, sphere-box, box-box contact detection, requires SYCL.
add_executable(test_core_narrowphase narrowphase_test.cpp)
target_link_libraries(test_core_narrowphase PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_narrowphase SOURCES narrowphase_test.cpp)
dyphur_add_smoke_test(test_core_narrowphase)

# XPBD solver test: position + velocity correction correctness, requires SYCL.
add_executable(test_core_xpbd xpbd_solver_test.cpp)
target_link_libraries(test_core_xpbd PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_xpbd SOURCES xpbd_solver_test.cpp)
dyphur_add_smoke_test(test_core_xpbd)

# Physics-level determinism test: small scene, 20 frames, hash vs golden.
add_executable(test_core_sim_determinism sim_determinism_test.cpp)
target_link_libraries(test_core_sim_determinism PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
if(DYPHUR_BACKEND STREQUAL "CUDA")
    target_compile_definitions(test_core_sim_determinism PRIVATE DYPHUR_BACKEND_IS_CUDA=1)
elseif(DYPHUR_BACKEND STREQUAL "HIP")
    target_compile_definitions(test_core_sim_determinism PRIVATE DYPHUR_BACKEND_IS_HIP=1)
endif()
add_sycl_to_target(TARGET test_core_sim_determinism SOURCES sim_determinism_test.cpp)
dyphur_add_determinism_test(test_core_sim_determinism)

# Ray-query test: BVH traversal + exact sphere/box intersection, requires SYCL.
add_executable(test_core_ray_query ray_query_test.cpp)
target_link_libraries(test_core_ray_query PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_ray_query SOURCES ray_query_test.cpp)
dyphur_add_smoke_test(test_core_ray_query)

# Per-stage microbenchmarks: integrator / broadphase / narrowphase / solver.
# Run with: test_core_bench --benchmark-samples 20 --reporter json > bench.json
add_executable(test_core_bench physics_benchmark.cpp)
target_link_libraries(test_core_bench PRIVATE
    dyphur::core
    Catch2::Catch2WithMain
)
add_sycl_to_target(TARGET test_core_bench SOURCES physics_benchmark.cpp)
# Benchmarks are not part of the smoke or determinism suites; run explicitly.
add_test(NAME test_core_bench COMMAND test_core_bench --benchmark-samples 5)
