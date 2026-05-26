#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <core/joint_store.hpp>
#include <compute/device.hpp>

using namespace dyphur;
using Catch::Matchers::WithinAbs;

static constexpr float kEps = 1e-5f;

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

    // Joint 0 checks
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

    // Joint 1 checks
    REQUIRE(bp[1] == 1u);
    REQUIRE(bc[1] == 2u);
    REQUIRE(type[1] == static_cast<uint8_t>(JointType::Ball));
}
