#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <core/body_store.hpp>
#include <compute/device.hpp>

using namespace dyphur;
using Catch::Matchers::WithinAbs;

static constexpr float kEps = 1e-5f;

TEST_CASE("BodyStore: construction and capacity", "[body][smoke]") {
    auto dev   = Device::default_cpu();
    auto s     = dev.make_stream();
    BodyStore store(s, 16);

    REQUIRE(store.count()    == 0);
    REQUIRE(store.capacity() == 16);
}

TEST_CASE("BodyStore: add and upload round-trip", "[body][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    BodyStore store(s, 8);

    // Body 0: dynamic at (1, 2, 3), moving at (0.5, 0, 0), mass 2
    {
        BodyParams p;
        p.position        = {1.f, 2.f, 3.f};
        p.linear_velocity = {0.5f, 0.f, 0.f};
        p.mass            = 2.f;
        p.inertia         = Mat3f::identity() * 4.f;  // I = 4*Id
        uint32_t id = store.add(p);
        REQUIRE(id == 0);
    }

    // Body 1: static at (0, 10, 0)
    {
        BodyParams p;
        p.position = {0.f, 10.f, 0.f};
        p.flags    = BodyFlag::Static;
        uint32_t id = store.add(p);
        REQUIRE(id == 1);
    }

    REQUIRE(store.count() == 2);

    store.upload();
    s.wait();

    // Download and verify via the raw device pointers in the view.
    BodyView v = store.view();
    REQUIRE(v.n == 2);

    std::vector<float>    pos_x(2), pos_y(2), pos_z(2);
    std::vector<float>    vel_x(2);
    std::vector<float>    inv_mass(2);
    std::vector<uint32_t> flags(2);

    s.queue().memcpy(pos_x.data(),    v.pos_x,    2 * sizeof(float)).wait();
    s.queue().memcpy(pos_y.data(),    v.pos_y,    2 * sizeof(float)).wait();
    s.queue().memcpy(pos_z.data(),    v.pos_z,    2 * sizeof(float)).wait();
    s.queue().memcpy(vel_x.data(),    v.vel_x,    2 * sizeof(float)).wait();
    s.queue().memcpy(inv_mass.data(), v.inv_mass, 2 * sizeof(float)).wait();
    s.queue().memcpy(flags.data(),    v.flags,    2 * sizeof(uint32_t)).wait();

    // Body 0
    REQUIRE_THAT(pos_x[0], WithinAbs(1.f, kEps));
    REQUIRE_THAT(pos_y[0], WithinAbs(2.f, kEps));
    REQUIRE_THAT(pos_z[0], WithinAbs(3.f, kEps));
    REQUIRE_THAT(vel_x[0], WithinAbs(0.5f, kEps));
    REQUIRE_THAT(inv_mass[0], WithinAbs(0.5f, kEps));   // 1/mass = 1/2
    REQUIRE((flags[0] & BodyFlag::Static) == 0u);

    // Body 1: static → inv_mass == 0
    REQUIRE_THAT(pos_y[1], WithinAbs(10.f, kEps));
    REQUIRE_THAT(inv_mass[1], WithinAbs(0.f, kEps));
    REQUIRE((flags[1] & BodyFlag::Static) != 0u);
}

TEST_CASE("BodyStore: inverse inertia stored correctly", "[body][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    BodyStore store(s, 4);

    // Diagonal inertia tensor diag(2, 4, 8) → inverse diag(0.5, 0.25, 0.125)
    BodyParams p;
    p.mass    = 1.f;
    p.inertia = Mat3f(2.f, 0.f, 0.f,
                      0.f, 4.f, 0.f,
                      0.f, 0.f, 8.f);
    store.add(p);
    store.upload();
    s.wait();

    BodyView v = store.view();
    float ixx, iyy, izz, ixy, ixz, iyz;
    s.queue().memcpy(&ixx, v.iI_xx, sizeof(float)).wait();
    s.queue().memcpy(&iyy, v.iI_yy, sizeof(float)).wait();
    s.queue().memcpy(&izz, v.iI_zz, sizeof(float)).wait();
    s.queue().memcpy(&ixy, v.iI_xy, sizeof(float)).wait();
    s.queue().memcpy(&ixz, v.iI_xz, sizeof(float)).wait();
    s.queue().memcpy(&iyz, v.iI_yz, sizeof(float)).wait();

    REQUIRE_THAT(ixx, WithinAbs(0.5f,   kEps));
    REQUIRE_THAT(iyy, WithinAbs(0.25f,  kEps));
    REQUIRE_THAT(izz, WithinAbs(0.125f, kEps));
    REQUIRE_THAT(ixy, WithinAbs(0.f, kEps));
    REQUIRE_THAT(ixz, WithinAbs(0.f, kEps));
    REQUIRE_THAT(iyz, WithinAbs(0.f, kEps));
}
