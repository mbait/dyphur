#include "core/math/math.hpp"
#include "core/body.hpp"
#include "core/articulation.hpp"
#include "core/realtime_enforcer.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <type_traits>
#include <cmath>
#include <thread>

using namespace dyphur;

// ── Vec3 ──────────────────────────────────────────────────────────────────────

TEST_CASE("Vec3: arithmetic", "[smoke]") {
    Vec3f a{1.f, 2.f, 3.f}, b{4.f, 5.f, 6.f};
    auto s = a + b;
    REQUIRE(s.x == 5.f); REQUIRE(s.y == 7.f); REQUIRE(s.z == 9.f);

    auto d = b - a;
    REQUIRE(d.x == 3.f); REQUIRE(d.y == 3.f); REQUIRE(d.z == 3.f);

    auto scaled = a * 2.f;
    REQUIRE(scaled.x == 2.f); REQUIRE(scaled.y == 4.f); REQUIRE(scaled.z == 6.f);
}

TEST_CASE("Vec3: dot and cross", "[smoke]") {
    Vec3f x = Vec3f::unit_x(), y = Vec3f::unit_y(), z = Vec3f::unit_z();
    REQUIRE(x.dot(y) == 0.f);
    REQUIRE(x.cross(y) == z);
    REQUIRE(y.cross(z) == x);
    REQUIRE(z.cross(x) == y);
}

TEST_CASE("Vec3: norm and normalize", "[smoke]") {
    Vec3f v{3.f, 4.f, 0.f};
    REQUIRE_THAT(v.norm(), Catch::Matchers::WithinULP(5.f, 1));
    Vec3f n = v.normalized();
    REQUIRE_THAT(n.norm(), Catch::Matchers::WithinAbs(1.f, 1e-6f));
}

// ── Mat3 ──────────────────────────────────────────────────────────────────────

TEST_CASE("Mat3: identity multiply", "[smoke]") {
    Mat3f I = Mat3f::identity();
    Vec3f v{1.f, 2.f, 3.f};
    Vec3f Iv = I * v;
    REQUIRE(Iv == v);
}

TEST_CASE("Mat3: transpose involution", "[smoke]") {
    Mat3f m{1,2,3, 4,5,6, 7,8,9};
    Mat3f tt = m.transpose().transpose();
    for (size_t i = 0; i < 9; ++i)
        REQUIRE(tt.data_[i] == m.data_[i]);
}

TEST_CASE("Mat3: skew-symmetric cross product", "[smoke]") {
    Vec3f a{1.f, 2.f, 3.f}, b{4.f, 5.f, 6.f};
    Vec3f cross_direct  = a.cross(b);
    Vec3f cross_via_mat = Mat3f::skew(a) * b;
    REQUIRE_THAT(cross_via_mat.x, Catch::Matchers::WithinULP(cross_direct.x, 0));
    REQUIRE_THAT(cross_via_mat.y, Catch::Matchers::WithinULP(cross_direct.y, 0));
    REQUIRE_THAT(cross_via_mat.z, Catch::Matchers::WithinULP(cross_direct.z, 0));
}

// ── Quat ─────────────────────────────────────────────────────────────────────

TEST_CASE("Quat: identity rotation", "[smoke]") {
    Quatf q = Quatf::identity();
    Vec3f v{1.f, 2.f, 3.f};
    Vec3f r = q.rotate(v);
    REQUIRE_THAT(r.x, Catch::Matchers::WithinULP(v.x, 1));
    REQUIRE_THAT(r.y, Catch::Matchers::WithinULP(v.y, 1));
    REQUIRE_THAT(r.z, Catch::Matchers::WithinULP(v.z, 1));
}

TEST_CASE("Quat: 90-degree rotation around Z", "[smoke]") {
    float angle = float(M_PI) / 2.f;
    Quatf q     = Quatf::from_axis_angle(Vec3f::unit_z(), angle);
    Vec3f r     = q.rotate(Vec3f::unit_x());
    REQUIRE_THAT(r.x, Catch::Matchers::WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT(r.y, Catch::Matchers::WithinAbs(1.f, 1e-6f));
    REQUIRE_THAT(r.z, Catch::Matchers::WithinAbs(0.f, 1e-6f));
}

TEST_CASE("Quat: composition equals double rotation", "[smoke]") {
    Vec3f axis  = Vec3f{1.f, 1.f, 0.f}.normalized();
    float angle = float(M_PI) / 4.f;
    Quatf q     = Quatf::from_axis_angle(axis, angle);
    Quatf q2    = q * q;
    Quatf qd    = Quatf::from_axis_angle(axis, 2.f * angle);
    REQUIRE_THAT(std::abs(q2.w - qd.w),         Catch::Matchers::WithinAbs(0.f, 1e-6f));
    REQUIRE_THAT((q2.xyz - qd.xyz).norm(), Catch::Matchers::WithinAbs(0.f, 1e-6f));
}

// ── Transform ────────────────────────────────────────────────────────────────

TEST_CASE("Transform: inverse round-trip", "[smoke]") {
    Vec3f p{1.f, 2.f, 3.f};
    Quatf r = Quatf::from_axis_angle(Vec3f::unit_y(), float(M_PI) / 3.f);
    Transformf T{p, r};
    Transformf Tinv = T.inverse();
    Vec3f pt{4.f, 5.f, 6.f};
    Vec3f rt = Tinv.transform_point(T.transform_point(pt));
    REQUIRE_THAT(rt.x, Catch::Matchers::WithinAbs(pt.x, 1e-5f));
    REQUIRE_THAT(rt.y, Catch::Matchers::WithinAbs(pt.y, 1e-5f));
    REQUIRE_THAT(rt.z, Catch::Matchers::WithinAbs(pt.z, 1e-5f));
}

// ── SpatialVector ────────────────────────────────────────────────────────────

TEST_CASE("SpatialVector: arithmetic", "[smoke]") {
    SpatialVectorf a{{1,2,3}, {4,5,6}};
    SpatialVectorf b{{1,1,1}, {2,2,2}};
    auto c = a + b;
    REQUIRE(c.angular.x == 2.f);
    REQUIRE(c.linear.z  == 8.f);
}

// ── Body ─────────────────────────────────────────────────────────────────────

TEST_CASE("BodyFlag: bitmask values are distinct and non-zero", "[smoke]") {
    REQUIRE(BodyFlag::Static    != 0u);
    REQUIRE(BodyFlag::Kinematic != 0u);
    REQUIRE(BodyFlag::Sleeping  != 0u);
    REQUIRE((BodyFlag::Static & BodyFlag::Kinematic) == 0u);
    REQUIRE((BodyFlag::Static & BodyFlag::Sleeping)  == 0u);
    REQUIRE((BodyFlag::Kinematic & BodyFlag::Sleeping) == 0u);
}

TEST_CASE("BodyParams: default construction", "[smoke]") {
    BodyParams p;
    REQUIRE(p.position        == Vec3f::zero());
    REQUIRE(p.rotation.w      == 1.f);
    REQUIRE(p.rotation.xyz    == Vec3f::zero());
    REQUIRE(p.linear_velocity == Vec3f::zero());
    REQUIRE(p.mass            == 1.f);
    REQUIRE(p.shape_handle    == 0u);
    REQUIRE(p.flags           == 0u);
}

TEST_CASE("BodyView: is trivially copyable", "[smoke]") {
    static_assert(std::is_trivially_copyable_v<BodyView>,
        "BodyView must be trivially copyable for safe kernel capture");
}

// ── Articulation ──────────────────────────────────────────────────────────────

TEST_CASE("JointType: enum values are distinct", "[smoke]") {
    REQUIRE(static_cast<uint8_t>(JointType::Fixed)    == 0);
    REQUIRE(static_cast<uint8_t>(JointType::Revolute) == 1);
    REQUIRE(static_cast<uint8_t>(JointType::Prismatic)== 2);
    REQUIRE(static_cast<uint8_t>(JointType::Ball)     == 3);
}

TEST_CASE("JointParams: default construction", "[smoke]") {
    JointParams p;
    REQUIRE(p.body_parent  == 0u);
    REQUIRE(p.body_child   == 0u);
    REQUIRE(p.anchor_parent == Vec3f::zero());
    REQUIRE(p.anchor_child  == Vec3f::zero());
    REQUIRE(p.limit_lo >  p.limit_hi);  // limits disabled by default
    REQUIRE(p.stiffness == 0.f);
    REQUIRE(p.damping   == 0.f);
    REQUIRE(p.type      == JointType::Fixed);
}

TEST_CASE("JointView: is trivially copyable", "[smoke]") {
    static_assert(std::is_trivially_copyable_v<JointView>,
        "JointView must be trivially copyable for safe kernel capture");
}

// ── RealtimeEnforcer ──────────────────────────────────────────────────────────

TEST_CASE("RealtimeEnforcer: first begin_frame returns dt_nominal", "[smoke]") {
    RealtimeEnforcer e;
    float dt = e.begin_frame();
    REQUIRE_THAT(dt, Catch::Matchers::WithinULP(1.f / 60.f, 0));
}

TEST_CASE("RealtimeEnforcer: dt is clamped to [dt_min, dt_max]", "[smoke]") {
    RealtimeEnforcer::Params p;
    p.dt_min = 1.f / 240.f;
    p.dt_max = 1.f / 15.f;
    RealtimeEnforcer e(p);

    // Frame 1: nominal
    e.begin_frame();
    e.end_frame();

    // Frame 2: wall time is very small (no sleep) → clamps to dt_min
    float dt = e.begin_frame();
    REQUIRE(dt >= p.dt_min);
    REQUIRE(dt <= p.dt_max);
}

TEST_CASE("RealtimeEnforcer: limit_realtime paces to dt_nominal", "[smoke]") {
    RealtimeEnforcer::Params p;
    p.dt_nominal     = 1.f / 60.f;
    p.limit_realtime = true;
    RealtimeEnforcer e(p);

    using Clock   = RealtimeEnforcer::Clock;
    using Seconds = RealtimeEnforcer::Seconds;

    e.begin_frame();
    auto t0     = Clock::now();
    e.end_frame();
    double wall = Seconds(Clock::now() - t0).count();

    // Should have slept to fill out ~1/60 s; allow ±5 ms tolerance.
    REQUIRE(wall >= p.dt_nominal - 0.005f);
}

TEST_CASE("RealtimeEnforcer: realtime_factor is positive after two frames", "[smoke]") {
    RealtimeEnforcer e;
    e.begin_frame();
    e.end_frame();
    e.begin_frame();
    e.end_frame();
    REQUIRE(e.realtime_factor() > 0.f);
}

TEST_CASE("RealtimeEnforcer: fast sim has realtime_factor >> 1", "[smoke]") {
    RealtimeEnforcer::Params p;
    p.limit_realtime = false;
    RealtimeEnforcer e(p);

    // Spin 10 frames with no work — should be much faster than realtime.
    for (int i = 0; i < 10; ++i) {
        e.begin_frame();
        e.end_frame();
    }
    // Running 60 Hz frames with zero work → wall << sim time → factor >> 1.
    REQUIRE(e.realtime_factor() > 1.f);
}
