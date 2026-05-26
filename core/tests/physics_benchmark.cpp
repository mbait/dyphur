#include <catch2/catch_test_macros.hpp>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <core/body_store.hpp>
#include <core/broadphase.hpp>
#include <core/integrator.hpp>
#include <core/narrowphase.hpp>
#include <core/shape_store.hpp>
#include <core/xpbd_solver.hpp>
#include <compute/device.hpp>

using namespace dyphur;

// Fixed scene shared across all benchmarks: 8×8×8 = 512 dynamic boxes + ground.
// Advance 5 frames first to get a populated contact set, then benchmark each stage.
namespace {

constexpr int NX = 8, NZ = 8, NY = 8;
constexpr int N_DYN = NX * NY * NZ;   // 512
constexpr int N_BODIES = N_DYN + 1;
constexpr float BOX_H = 0.4f;
constexpr float GND_HY = 0.5f, GND_HX = 12.f;
constexpr float DT = 1.f / 60.f;
constexpr uint32_t MAX_PAIRS = 32768u, MAX_CONTACTS = 8192u;

struct BenchScene {
    Device      dev;
    Stream      s;
    ShapeStore  ss;
    BodyStore   bs;
    Broadphase  bp;
    Narrowphase np;
    XpbdSolver  solver;
    IntegratorParams ip;
    BodyView    bv;
    ShapeView   sv;
    AABB        bounds;

    explicit BenchScene()
        : dev(Device::default_cpu())
        , s(dev.make_stream())
        , ss(s, 4)
        , bs(s, N_BODIES)
        , bp(s, N_BODIES, MAX_PAIRS)
        , np(s, MAX_CONTACTS)
        , solver(s, 10)
    {
        ShapeParams dsp;
        dsp.type = ShapeType::Box; dsp.half_x = dsp.half_y = dsp.half_z = BOX_H;
        uint32_t ds = ss.add(dsp);
        ShapeParams gsp;
        gsp.type = ShapeType::Box; gsp.half_x = gsp.half_z = GND_HX; gsp.half_y = GND_HY;
        uint32_t gs = ss.add(gsp);
        ss.upload();

        const float m = 1.f, Idiag = m / 3.f * 2.f * BOX_H * BOX_H;
        BodyParams dp;
        dp.mass = m;
        dp.inertia = Mat3f(Idiag, 0, 0, 0, Idiag, 0, 0, 0, Idiag);
        dp.shape_handle = ds; dp.flags = 0;
        for (int iy = 0; iy < NY; ++iy)
            for (int ix = 0; ix < NX; ++ix)
                for (int iz = 0; iz < NZ; ++iz) {
                    dp.position = { (ix - NX/2 + 0.5f) * 1.f,
                                    GND_HY + (iy + 1) * 1.f,
                                    (iz - NZ/2 + 0.5f) * 1.f };
                    bs.add(dp);
                }
        BodyParams gp; gp.mass = 0; gp.position = {0, 0, 0};
        gp.shape_handle = gs; gp.flags = BodyFlag::Static;
        bs.add(gp);
        bs.upload(); s.wait();

        ip.gravity = {0, -9.81f, 0}; ip.dt = DT; ip.n_substeps = 1;
        bv = bs.view(); sv = ss.view();
        bounds = {-15, -2, -15, 15, 20, 15};

        // Advance 5 frames to populate contacts before benchmarking.
        for (int f = 0; f < 5; ++f) {
            integrate(s, bv, ip); s.wait();
            bp.build_and_query(s, bv, sv, bounds); s.wait();
            uint32_t np_ = bp.download_count(s);
            bp.sort_pairs(s, np_); s.wait();
            np.run(s, bp.pairs_ptr(), np_, bv, sv); s.wait();
            solver.solve(s, np.contacts(), bv, DT); s.wait();
        }
    }
};

// Global scene — constructed once for the whole benchmark run.
static BenchScene& scene() {
    static BenchScene sc;
    return sc;
}

} // namespace

TEST_CASE("physics benchmarks: 512 bodies", "[benchmark]") {
    auto& sc = scene();

    BENCHMARK("integrator") {
        integrate(sc.s, sc.bv, sc.ip);
        sc.s.wait();
    };

    BENCHMARK("broadphase build+query") {
        sc.bp.build_and_query(sc.s, sc.bv, sc.sv, sc.bounds);
        sc.s.wait();
    };

    // Re-run broadphase to get a stable pair count for subsequent stages.
    sc.bp.build_and_query(sc.s, sc.bv, sc.sv, sc.bounds);
    sc.s.wait();
    uint32_t n_pairs = sc.bp.download_count(sc.s);

    BENCHMARK("broadphase sort_pairs") {
        sc.bp.sort_pairs(sc.s, n_pairs);
        sc.s.wait();
    };

    sc.bp.sort_pairs(sc.s, n_pairs);
    sc.s.wait();

    BENCHMARK("narrowphase") {
        sc.np.run(sc.s, sc.bp.pairs_ptr(), n_pairs, sc.bv, sc.sv);
        sc.s.wait();
    };

    sc.np.run(sc.s, sc.bp.pairs_ptr(), n_pairs, sc.bv, sc.sv);
    sc.s.wait();

    BENCHMARK("xpbd solver") {
        sc.solver.solve(sc.s, sc.np.contacts(), sc.bv, DT);
        sc.s.wait();
    };
}
