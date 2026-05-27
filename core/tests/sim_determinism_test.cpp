#include <catch2/catch_test_macros.hpp>
#include <core/body_store.hpp>
#include <core/broadphase.hpp>
#include <core/integrator.hpp>
#include <core/narrowphase.hpp>
#include <core/shape_store.hpp>
#include <core/xpbd_solver.hpp>
#include <core/articulation.hpp>
#include <core/contact_store.hpp>
#include <core/joint_store.hpp>
#include <compute/device.hpp>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace dyphur;

// Scene: 2×2×2 = 8 dynamic boxes + 1 static ground, 20 frames at 60 Hz.
// Golden hash is backend-specific; the test is registered per-backend and the
// golden value encodes the hardware+build combination.
namespace {

constexpr int NX = 2, NZ = 2, NY = 2;
constexpr int N_DYN = NX * NY * NZ;   // 8
constexpr int N_BODIES = N_DYN + 1;   // +1 static ground
constexpr float BOX_H = 0.4f;
constexpr float GND_HY = 0.5f, GND_HX = 5.f;
constexpr float DT = 1.f / 60.f;
constexpr int N_FRAMES = 20;
constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;

static uint64_t fnv1a64(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x00000100000001B3ULL; }
    return h;
}

static uint64_t run_sim(Device& dev) {
    auto s  = dev.make_stream();
    auto& q = s.queue();

    ShapeStore ss(s, 4);
    ShapeParams dsp;
    dsp.type = ShapeType::Box; dsp.half_x = dsp.half_y = dsp.half_z = BOX_H;
    uint32_t ds = ss.add(dsp);
    ShapeParams gsp;
    gsp.type = ShapeType::Box; gsp.half_x = gsp.half_z = GND_HX; gsp.half_y = GND_HY;
    uint32_t gs = ss.add(gsp);
    ss.upload();

    BodyStore bs(s, N_BODIES);
    const float m = 1.f, Idiag = m / 3.f * 2.f * BOX_H * BOX_H;
    BodyParams dp;
    dp.mass = m;
    dp.inertia = Mat3f(Idiag, 0, 0, 0, Idiag, 0, 0, 0, Idiag);
    dp.shape_handle = ds;
    dp.flags = 0;
    for (int iy = 0; iy < NY; ++iy)
        for (int ix = 0; ix < NX; ++ix)
            for (int iz = 0; iz < NZ; ++iz) {
                dp.position = { (ix - NX/2 + 0.5f) * 1.f,
                                GND_HY + (iy + 1) * 1.f,
                                (iz - NZ/2 + 0.5f) * 1.f };
                bs.add(dp);
            }
    BodyParams gp;
    gp.mass = 0; gp.position = {0, 0, 0};
    gp.shape_handle = gs; gp.flags = BodyFlag::Static;
    bs.add(gp);
    bs.upload(); s.wait();

    Broadphase  bp(s, N_BODIES, 4096u);
    Narrowphase np(s, 1024u);
    XpbdSolver  solver(s, 10);

    IntegratorParams ip;
    ip.gravity = {0, -9.81f, 0}; ip.dt = DT; ip.n_substeps = 1;

    BodyView bv = bs.view(); ShapeView sv = ss.view();
    const AABB bounds = {-10, -2, -10, 10, 15, 10};

    for (int f = 0; f < N_FRAMES; ++f) {
        integrate(s, bv, ip); s.wait();
        bp.build_and_query(s, bv, sv, bounds); s.wait();
        uint32_t np_ = bp.download_count(s);
        bp.sort_pairs(s, np_); s.wait();
        np.run(s, bp.pairs_ptr(), np_, bv, sv); s.wait();
        solver.solve(s, np.contacts(), JointView{}, bv, DT); s.wait();
    }

    std::vector<float> hx(N_BODIES), hy(N_BODIES), hz(N_BODIES);
    std::vector<float> hw(N_BODIES), hqx(N_BODIES), hqy(N_BODIES), hqz(N_BODIES);
    q.memcpy(hx.data(),  bv.pos_x, N_BODIES * sizeof(float));
    q.memcpy(hy.data(),  bv.pos_y, N_BODIES * sizeof(float));
    q.memcpy(hz.data(),  bv.pos_z, N_BODIES * sizeof(float));
    q.memcpy(hw.data(),  bv.rot_w, N_BODIES * sizeof(float));
    q.memcpy(hqx.data(), bv.rot_x, N_BODIES * sizeof(float));
    q.memcpy(hqy.data(), bv.rot_y, N_BODIES * sizeof(float));
    q.memcpy(hqz.data(), bv.rot_z, N_BODIES * sizeof(float)).wait();

    uint64_t h = FNV_OFFSET;
    h = fnv1a64(h, hx.data(),  N_BODIES * sizeof(float));
    h = fnv1a64(h, hy.data(),  N_BODIES * sizeof(float));
    h = fnv1a64(h, hz.data(),  N_BODIES * sizeof(float));
    h = fnv1a64(h, hw.data(),  N_BODIES * sizeof(float));
    h = fnv1a64(h, hqx.data(), N_BODIES * sizeof(float));
    h = fnv1a64(h, hqy.data(), N_BODIES * sizeof(float));
    h = fnv1a64(h, hqz.data(), N_BODIES * sizeof(float));
    return h;
}

} // namespace

// ── Articulation determinism helper ──────────────────────────────────────────
//
// Scene: static base + 2 dynamic links connected by revolute joints.
// PD motors drive both joints toward non-zero targets; gravity is on.
// 20 frames at 60 Hz.  No broadphase/narrowphase (tests joint pipeline only).

static uint64_t run_artic_sim(Device& dev) {
    auto s  = dev.make_stream();
    auto& q = s.queue();

    constexpr int   N_AB    = 3;          // base + 2 links
    constexpr float LK_HY   = 0.15f;
    constexpr float LK_MASS = 1.f;
    constexpr float SEP     = 2.f * LK_HY;

    BodyStore bs(s, N_AB);

    // Static base.
    { BodyParams p; p.position = {0,0,0}; p.mass = 0; p.flags = BodyFlag::Static;
      p.inertia = Mat3f::identity(); bs.add(p); }

    // Two dynamic links.
    const float Idiag = LK_MASS / 3.f * 2.f * LK_HY * LK_HY;
    const Mat3f lI(Idiag,0,0, 0,Idiag,0, 0,0,Idiag);
    for (int i = 1; i <= 2; ++i) {
        BodyParams p;
        p.position = {0.f, static_cast<float>(i) * SEP, 0.f};
        p.mass = LK_MASS; p.inertia = lI; p.flags = 0;
        bs.add(p);
    }
    bs.upload(); s.wait();

    JointStore js(s, 2);
    const float targets[2] = {0.5f, -0.5f};
    for (int i = 0; i < 2; ++i) {
        JointParams p;
        p.body_parent   = static_cast<uint32_t>(i);
        p.body_child    = static_cast<uint32_t>(i + 1);
        p.anchor_parent = {0.f, LK_HY, 0.f};
        p.anchor_child  = {0.f, -LK_HY, 0.f};
        p.axis          = {1.f, 0.f, 0.f};     // revolute around X
        p.limit_lo      = -2.9f; p.limit_hi = 2.9f;
        p.target_pos    = targets[i];
        p.target_vel    = 0.f;
        p.stiffness     = 200.f; p.damping = 30.f;
        p.compliance_pos = 0.f; p.compliance_ang = 0.f;
        p.type          = JointType::Revolute;
        js.add(p);
    }
    js.upload(); s.wait();

    ContactStore cs(s, 1); cs.reset(s); s.wait();

    XpbdSolver solver(s, 10);
    IntegratorParams ip; ip.gravity = {0,-9.81f,0}; ip.dt = DT; ip.n_substeps = 1;

    BodyView  bv = bs.view();
    JointView jv = js.view();
    for (int f = 0; f < N_FRAMES; ++f) {
        integrate(s, bv, ip); s.wait();
        solver.solve(s, cs.view(), jv, bv, DT); s.wait();
    }

    std::vector<float> hx(N_AB), hy(N_AB), hz(N_AB);
    std::vector<float> hw(N_AB), hqx(N_AB), hqy(N_AB), hqz(N_AB);
    q.memcpy(hx.data(),  bv.pos_x, N_AB * sizeof(float));
    q.memcpy(hy.data(),  bv.pos_y, N_AB * sizeof(float));
    q.memcpy(hz.data(),  bv.pos_z, N_AB * sizeof(float));
    q.memcpy(hw.data(),  bv.rot_w, N_AB * sizeof(float));
    q.memcpy(hqx.data(), bv.rot_x, N_AB * sizeof(float));
    q.memcpy(hqy.data(), bv.rot_y, N_AB * sizeof(float));
    q.memcpy(hqz.data(), bv.rot_z, N_AB * sizeof(float)).wait();

    uint64_t h = FNV_OFFSET;
    h = fnv1a64(h, hx.data(),  N_AB * sizeof(float));
    h = fnv1a64(h, hy.data(),  N_AB * sizeof(float));
    h = fnv1a64(h, hz.data(),  N_AB * sizeof(float));
    h = fnv1a64(h, hw.data(),  N_AB * sizeof(float));
    h = fnv1a64(h, hqx.data(), N_AB * sizeof(float));
    h = fnv1a64(h, hqy.data(), N_AB * sizeof(float));
    h = fnv1a64(h, hqz.data(), N_AB * sizeof(float));
    return h;
}

// Run the scene twice on CPU and assert both hashes match the golden value.
// The golden value encodes the physics state for this build; any change to
// the physics pipeline that alters the trajectory will fail this test.
// To re-derive: run the test binary and read the INFO line on failure (or
// temporarily set GOLDEN = 0 to skip the equality check and print the hash).
TEST_CASE("sim determinism: 8 boxes + ground, 20 frames", "[determinism]") {
    // CPU (OMP) golden — must be re-derived if physics logic changes.
    constexpr uint64_t GOLDEN = 0xcbcd209c3665c818ULL;

    Device dev = Device::default_cpu();

    uint64_t h1 = run_sim(dev);
    uint64_t h2 = run_sim(dev);

    // Run-to-run consistency is the core guarantee.
    REQUIRE(h1 == h2);

    // Golden check — skipped while GOLDEN is placeholder zero.
    if (GOLDEN != 0) {
        INFO("hash=0x" << std::hex << h1 << " golden=0x" << GOLDEN);
        REQUIRE(h1 == GOLDEN);
    }
}

TEST_CASE("sim determinism: 2-link articulated arm, 20 frames", "[determinism]") {
    // CPU (OMP) golden — must be re-derived if physics logic changes.
    constexpr uint64_t GOLDEN = 0xa417e4fb155bea78ULL;

    Device dev = Device::default_cpu();

    uint64_t h1 = run_artic_sim(dev);
    uint64_t h2 = run_artic_sim(dev);

    INFO("artic hash=0x" << std::hex << h1);
    REQUIRE(h1 == h2);

    if (GOLDEN != 0) {
        INFO("hash=0x" << std::hex << h1 << " golden=0x" << GOLDEN);
        REQUIRE(h1 == GOLDEN);
    }
}
