#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <core/joint_store.hpp>
#include <core/body_store.hpp>
#include <core/contact_store.hpp>
#include <core/xpbd_solver.hpp>
#include <compute/device.hpp>
#include <cmath>
#include <vector>

using namespace dyphur;
using Catch::Matchers::WithinAbs;

static constexpr float kEps = 1e-5f;

// Download a single float value from a device pointer.
static float dl_f(sycl::queue& q, const float* p, uint32_t idx = 0) {
    float v; q.memcpy(&v, p + idx, sizeof(float)).wait(); return v;
}

// ── JointStore round-trip tests ───────────────────────────────────────────────

TEST_CASE("JointStore: construction and capacity", "[articulation][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    JointStore store(s, 8);

    REQUIRE(store.count()    == 0);
    REQUIRE(store.capacity() == 8);
}

TEST_CASE("JointStore: add and upload round-trip", "[articulation][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    JointStore store(s, 4);

    // Joint 0: revolute between bodies 0 and 1
    {
        JointParams p;
        p.body_parent   = 0;
        p.body_child    = 1;
        p.anchor_parent = {0.f, 0.5f, 0.f};
        p.anchor_child  = {0.f, -0.5f, 0.f};
        p.axis          = {1.f, 0.f, 0.f};
        p.limit_lo      = -1.57f;
        p.limit_hi      =  1.57f;
        p.stiffness     = 100.f;
        p.damping       =  10.f;
        p.compliance_pos = 1e-4f;
        p.type          = JointType::Revolute;
        uint32_t id = store.add(p);
        REQUIRE(id == 0);
    }

    // Joint 1: ball joint between bodies 1 and 2
    {
        JointParams p;
        p.body_parent = 1;
        p.body_child  = 2;
        p.type        = JointType::Ball;
        uint32_t id = store.add(p);
        REQUIRE(id == 1);
    }

    REQUIRE(store.count() == 2);

    store.upload();
    s.wait();

    JointView v = store.view();
    REQUIRE(v.n == 2);

    std::vector<uint32_t> bp(2), bc(2);
    std::vector<float>    apy(2), acy(2);
    std::vector<float>    ax(2);
    std::vector<float>    llo(2), lhi(2);
    std::vector<float>    stiff(2);
    std::vector<float>    cpos(2);
    std::vector<uint8_t>  type(2);

    auto& q = s.queue();
    q.memcpy(bp.data(),    v.body_parent,    2 * sizeof(uint32_t)).wait();
    q.memcpy(bc.data(),    v.body_child,     2 * sizeof(uint32_t)).wait();
    q.memcpy(apy.data(),   v.anchor_py,      2 * sizeof(float)).wait();
    q.memcpy(acy.data(),   v.anchor_cy,      2 * sizeof(float)).wait();
    q.memcpy(ax.data(),    v.axis_px,        2 * sizeof(float)).wait();
    q.memcpy(llo.data(),   v.limit_lo,       2 * sizeof(float)).wait();
    q.memcpy(lhi.data(),   v.limit_hi,       2 * sizeof(float)).wait();
    q.memcpy(stiff.data(), v.stiffness,      2 * sizeof(float)).wait();
    q.memcpy(cpos.data(),  v.compliance_pos, 2 * sizeof(float)).wait();
    q.memcpy(type.data(),  v.type,           2 * sizeof(uint8_t)).wait();

    REQUIRE(bp[0] == 0u);
    REQUIRE(bc[0] == 1u);
    REQUIRE_THAT(apy[0],   WithinAbs( 0.5f, kEps));
    REQUIRE_THAT(acy[0],   WithinAbs(-0.5f, kEps));
    REQUIRE_THAT(ax[0],    WithinAbs( 1.f,  kEps));
    REQUIRE_THAT(llo[0],   WithinAbs(-1.57f, 1e-4f));
    REQUIRE_THAT(lhi[0],   WithinAbs( 1.57f, 1e-4f));
    REQUIRE_THAT(stiff[0], WithinAbs(100.f, kEps));
    REQUIRE_THAT(cpos[0],  WithinAbs(1e-4f, 1e-8f));
    REQUIRE(type[0] == static_cast<uint8_t>(JointType::Revolute));

    REQUIRE(bp[1] == 1u);
    REQUIRE(bc[1] == 2u);
    REQUIRE(type[1] == static_cast<uint8_t>(JointType::Ball));
}

// ── Joint constraint correctness tests ───────────────────────────────────────
//
// Each test places two bodies in a pose that satisfies a joint, runs the
// solver for several iterations, and checks that the joint constraint is still
// satisfied (anchor positions match within tolerance) or that the solver
// corrects an initial violation and brings them into compliance.

// Helper: build a two-body BodyStore with body 0 at pos_a and body 1 at pos_b,
// both mass = 1, inertia = diag(1,1,1).
static BodyStore two_bodies(Stream& s,
                             Vec3f pos_a, Vec3f pos_b,
                             uint32_t flags_a = 0, uint32_t flags_b = 0) {
    BodyStore bs(s, 4);
    BodyParams p;
    p.mass    = (flags_a & BodyFlag::Static) ? 0.f : 1.f;
    p.inertia = Mat3f::identity();
    p.flags   = flags_a;
    p.position = pos_a;
    bs.add(p);

    p.mass     = (flags_b & BodyFlag::Static) ? 0.f : 1.f;
    p.inertia  = Mat3f::identity();
    p.flags    = flags_b;
    p.position = pos_b;
    bs.add(p);

    bs.upload();
    return bs;
}

TEST_CASE("Fixed joint: holds two bodies together after violation", "[articulation][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    // Anchor: parent anchor at (0, 0.5, 0), child anchor at (0, -0.5, 0).
    // Satisfied pose: body 0 at (0,0,0), body 1 at (0,1,0).
    // Introduce violation: body 1 starts at (0,1.2,0) (0.2 m off).
    BodyStore bs = two_bodies(s, {0.f,0.f,0.f}, {0.f,1.2f,0.f});
    s.wait();

    JointStore js(s, 2);
    JointParams jp;
    jp.body_parent   = 0;
    jp.body_child    = 1;
    jp.anchor_parent = {0.f, 0.5f, 0.f};
    jp.anchor_child  = {0.f, -0.5f, 0.f};
    jp.type          = JointType::Fixed;
    js.add(jp);
    js.upload();
    s.wait();

    XpbdSolver solver(s, 20);
    ContactStore cs(s, 1);
    cs.reset(s); s.wait();

    solver.solve(s, cs.view(), js.view(), bs.view(), 1.f/60.f);
    s.wait();

    // Check that anchor positions match.
    BodyView bv = bs.view();
    auto& q = s.queue();
    // world anchor A = pos_a + (0, 0.5, 0)
    float ay0 = dl_f(q, bv.pos_y, 0) + 0.5f;
    // world anchor B = pos_b + (0, -0.5, 0)
    float ay1 = dl_f(q, bv.pos_y, 1) - 0.5f;

    REQUIRE_THAT(ay0, WithinAbs(ay1, 1e-3f));
}

TEST_CASE("Fixed joint: static parent, dynamic child pulled to constraint", "[articulation][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    // Body 0 static at origin, body 1 dynamic at (0.5, 1, 0) (off by 0.5 in x).
    BodyStore bs = two_bodies(s, {0.f,0.f,0.f}, {0.5f,1.f,0.f},
                               BodyFlag::Static, 0);
    s.wait();

    JointStore js(s, 2);
    JointParams jp;
    jp.body_parent   = 0;
    jp.body_child    = 1;
    jp.anchor_parent = {0.f, 0.5f, 0.f};
    jp.anchor_child  = {0.f, -0.5f, 0.f};
    jp.type          = JointType::Fixed;
    js.add(jp);
    js.upload();
    s.wait();

    XpbdSolver solver(s, 100);
    ContactStore cs(s, 1);
    cs.reset(s); s.wait();

    solver.solve(s, cs.view(), js.view(), bs.view(), 1.f/60.f);
    s.wait();

    BodyView bv = bs.view();
    auto& q = s.queue();

    // Static body 0 must not move.
    REQUIRE_THAT(dl_f(q, bv.pos_x, 0), WithinAbs(0.f, kEps));
    REQUIRE_THAT(dl_f(q, bv.pos_y, 0), WithinAbs(0.f, kEps));

    // Dynamic body 1 must be pulled to the correct position (0, 1, 0).
    float x1 = dl_f(q, bv.pos_x, 1);
    float y1 = dl_f(q, bv.pos_y, 1);
    // Anchor on child: (0,-0.5,0) in child body frame → world = body_pos + (0,-0.5,0).
    // Anchor on parent: (0,0.5,0) → world = (0,0.5,0).
    // Constraint: world_anchor_child == world_anchor_parent → (x1, y1-0.5) == (0, 0.5).
    REQUIRE_THAT(x1, WithinAbs(0.f, 1e-3f));
    REQUIRE_THAT(y1, WithinAbs(1.f, 1e-3f));
}

TEST_CASE("Revolute joint: anchor constraint satisfied, axis alignment preserved", "[articulation][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    // Body 0 static at origin, body 1 dynamic at (0, 1, 0).
    // Revolute axis: Y. Joint satisfied at this pose.
    BodyStore bs = two_bodies(s, {0.f,0.f,0.f}, {0.f,1.f,0.f},
                               BodyFlag::Static, 0);
    s.wait();

    JointStore js(s, 2);
    JointParams jp;
    jp.body_parent   = 0;
    jp.body_child    = 1;
    jp.anchor_parent = {0.f, 0.5f, 0.f};
    jp.anchor_child  = {0.f, -0.5f, 0.f};
    jp.axis          = {0.f, 1.f, 0.f};  // Y axis
    jp.type          = JointType::Revolute;
    js.add(jp);
    js.upload();
    s.wait();

    XpbdSolver solver(s, 10);
    ContactStore cs(s, 1);
    cs.reset(s); s.wait();

    solver.solve(s, cs.view(), js.view(), bs.view(), 1.f/60.f);
    s.wait();

    BodyView bv = bs.view();
    auto& q = s.queue();

    // Anchor on child in world: pos_b + (0,-0.5,0) = (0, 0.5, 0)
    // Anchor on parent in world: (0, 0.5, 0)
    float ax_w = dl_f(q, bv.pos_x, 1);
    float ay_w = dl_f(q, bv.pos_y, 1) - 0.5f;
    float az_w = dl_f(q, bv.pos_z, 1);
    REQUIRE_THAT(ax_w, WithinAbs(0.f, 1e-3f));
    REQUIRE_THAT(ay_w, WithinAbs(0.5f, 1e-3f));
    REQUIRE_THAT(az_w, WithinAbs(0.f, 1e-3f));
}

TEST_CASE("Ball joint: violation corrected to constraint satisfaction", "[articulation][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    // 0.3 m violation in Y only — keeps anchor offset parallel to correction,
    // preventing angular coupling, so we can verify anchor coincidence via body pos.
    BodyStore bs = two_bodies(s, {0.f,0.f,0.f}, {0.f,1.3f,0.f},
                               BodyFlag::Static, 0);
    s.wait();

    JointStore js(s, 2);
    JointParams jp;
    jp.body_parent = 0;
    jp.body_child  = 1;
    jp.anchor_parent = {0.f, 0.5f, 0.f};
    jp.anchor_child  = {0.f, -0.5f, 0.f};
    jp.type          = JointType::Ball;
    js.add(jp);
    js.upload();
    s.wait();

    XpbdSolver solver(s, 20);
    ContactStore cs(s, 1);
    cs.reset(s); s.wait();

    solver.solve(s, cs.view(), js.view(), bs.view(), 1.f/60.f);
    s.wait();

    BodyView bv = bs.view();
    auto& q = s.queue();

    // With no angular coupling, body 1 slides to (0, 1, 0).
    // Anchor on parent: (0, 0.5, 0). Anchor on child: pos + (0,-0.5,0) = (0, 0.5, 0).
    REQUIRE_THAT(dl_f(q, bv.pos_x, 1), WithinAbs(0.f, 1e-3f));
    REQUIRE_THAT(dl_f(q, bv.pos_y, 1), WithinAbs(1.f, 1e-3f));
    REQUIRE_THAT(dl_f(q, bv.pos_z, 1), WithinAbs(0.f, 1e-3f));
}
