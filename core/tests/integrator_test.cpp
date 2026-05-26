#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <core/body_store.hpp>
#include <core/integrator.hpp>
#include <compute/device.hpp>
#include <cmath>

using namespace dyphur;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// Download one scalar from a device pointer (blocking).
static float get1(sycl::queue& q, const float* ptr) {
    float v;
    q.memcpy(&v, ptr, sizeof(float)).wait();
    return v;
}
static uint32_t get1u(sycl::queue& q, const uint32_t* ptr) {
    uint32_t v;
    q.memcpy(&v, ptr, sizeof(uint32_t)).wait();
    return v;
}

// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("Integrator: static body does not move", "[integrator][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    BodyStore store(s, 4);

    BodyParams p;
    p.position = {1.f, 2.f, 3.f};
    p.flags    = BodyFlag::Static;
    store.add(p);
    store.upload();
    s.wait();

    IntegratorParams ip;
    ip.gravity    = {0.f, -9.81f, 0.f};
    ip.dt         = 1.f / 60.f;
    ip.n_substeps = 1;

    integrate(s, store.view(), ip);
    s.wait();

    BodyView v = store.view();
    REQUIRE_THAT(get1(s.queue(), v.pos_x), WithinAbs(1.f, 1e-6f));
    REQUIRE_THAT(get1(s.queue(), v.pos_y), WithinAbs(2.f, 1e-6f));
    REQUIRE_THAT(get1(s.queue(), v.pos_z), WithinAbs(3.f, 1e-6f));
}

TEST_CASE("Integrator: free fall matches symplectic Euler formula", "[integrator][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    BodyStore store(s, 4);

    BodyParams p;
    p.position = {0.f, 10.f, 0.f};
    p.mass     = 1.f;
    p.inertia  = Mat3f::identity();
    store.add(p);
    store.upload();
    s.wait();

    // Single step of dt=1 s, N=60 substeps → total_substeps=60.
    // Symplectic Euler exact formula for N substeps of sub_dt=1/N, from rest:
    //   y_N = y0 + g * (N+1) / (2*N)
    // (The continuous limit g/2 is only recovered as N→∞.)
    IntegratorParams ip;
    ip.gravity    = {0.f, -9.81f, 0.f};
    ip.dt         = 1.f;
    ip.n_substeps = 60;

    integrate(s, store.view(), ip);
    s.wait();

    const int   N          = 60;
    const float expected_y = 10.f + (-9.81f) * float(N + 1) / float(2 * N);
    BodyView v = store.view();
    REQUIRE_THAT(get1(s.queue(), v.pos_y), WithinAbs(expected_y, 1e-3f));
    REQUIRE_THAT(get1(s.queue(), v.pos_x), WithinAbs(0.f, 1e-5f));
    REQUIRE_THAT(get1(s.queue(), v.pos_z), WithinAbs(0.f, 1e-5f));
}

TEST_CASE("Integrator: constant velocity, no gravity", "[integrator][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    BodyStore store(s, 4);

    BodyParams p;
    p.position        = {0.f, 0.f, 0.f};
    p.linear_velocity = {3.f, 0.f, 0.f};
    p.mass            = 1.f;
    p.inertia         = Mat3f::identity();
    store.add(p);
    store.upload();
    s.wait();

    IntegratorParams ip;
    ip.gravity    = {0.f, 0.f, 0.f};
    ip.dt         = 1.f;
    ip.n_substeps = 1;

    integrate(s, store.view(), ip);
    s.wait();

    BodyView v = store.view();
    // x = 0 + 3 * 1 = 3
    REQUIRE_THAT(get1(s.queue(), v.pos_x), WithinAbs(3.f, 1e-5f));
    REQUIRE_THAT(get1(s.queue(), v.pos_y), WithinAbs(0.f, 1e-5f));
}

TEST_CASE("Integrator: quaternion stays normalised under spin", "[integrator][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    BodyStore store(s, 4);

    BodyParams p;
    p.mass            = 1.f;
    p.inertia         = Mat3f::identity();
    p.angular_velocity= {0.f, 1.f, 0.f};  // 1 rad/s around Y
    store.add(p);
    store.upload();
    s.wait();

    IntegratorParams ip;
    ip.gravity    = {0.f, 0.f, 0.f};
    ip.dt         = 1.f / 60.f;
    ip.n_substeps = 4;

    // Run for 1 simulated second (60 steps).
    for (int i = 0; i < 60; ++i)
        integrate(s, store.view(), ip);
    s.wait();

    BodyView v = store.view();
    float qw = get1(s.queue(), v.rot_w);
    float qx = get1(s.queue(), v.rot_x);
    float qy = get1(s.queue(), v.rot_y);
    float qz = get1(s.queue(), v.rot_z);
    float norm_sq = qw*qw + qx*qx + qy*qy + qz*qz;
    REQUIRE_THAT(norm_sq, WithinAbs(1.f, 1e-4f));
    // Y component must be non-zero (rotation happened).
    REQUIRE(std::abs(qy) > 0.1f);
}
