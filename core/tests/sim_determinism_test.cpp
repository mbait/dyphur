#include <catch2/catch_test_macros.hpp>
#include <core/body_store.hpp>
#include <core/broadphase.hpp>
#include <core/integrator.hpp>
#include <core/narrowphase.hpp>
#include <core/shape_store.hpp>
#include <core/xpbd_solver.hpp>
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
        solver.solve(s, np.contacts(), bv, DT); s.wait();
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
