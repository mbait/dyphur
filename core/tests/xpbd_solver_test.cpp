#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <core/body_store.hpp>
#include <core/narrowphase.hpp>
#include <core/shape_store.hpp>
#include <core/xpbd_solver.hpp>
#include <core/articulation.hpp>
#include <compute/buffer.hpp>
#include <compute/device.hpp>
#include <vector>

using namespace dyphur;
using Catch::Matchers::WithinAbs;

static constexpr float kEps = 1e-3f;

// Download a single float field from a device pointer.
static float download_f(sycl::queue& q, const float* d_ptr) {
    float val;
    q.memcpy(&val, d_ptr, sizeof(float)).wait();
    return val;
}

// Create a pairs buffer with one pair.
static Buffer<ContactPair> one_pair(Stream& s, uint32_t a, uint32_t b) {
    Buffer<ContactPair> buf(s, 1);
    ContactPair p{a, b};
    buf.upload(&p, 1);
    s.wait();
    return buf;
}

// ─────────────────────────────────────────────────────────────────────────────

// Two spheres (r=1) at (0,0,0) and (0,1.5,0): depth=0.5.
// After one solver pass both should be separated (dist >= 2.0).
TEST_CASE("XpbdSolver: sphere-sphere separates", "[solver][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp; sp.type = ShapeType::Sphere; sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 2);
    BodyParams p; p.mass = 1.f;
    p.position = {0.f, 0.f, 0.f}; bs.add(p);
    p.position = {0.f, 1.5f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    auto pbuf = one_pair(s, 0u, 1u);
    Narrowphase np(s, 8);
    np.run(s, pbuf.data(), 1u, bs.view(), ss.view());
    s.wait();

    REQUIRE(np.download_count(s) == 1u);

    XpbdSolver solver(s, 1);
    solver.solve(s, np.contacts(), JointView{}, bs.view(), 1.f/60.f);
    s.wait();

    auto& q = s.queue();
    float y0 = download_f(q, bs.view().pos_y);
    float y1 = download_f(q, bs.view().pos_y + 1);

    // Separation must be >= sum_r = 2.0 (to within tolerance).
    REQUIRE(y1 - y0 >= 2.f - kEps);
}

// Two equal-mass spheres, one approaching the other.
// After solve: normal relative velocity must be >= 0 (no longer approaching).
TEST_CASE("XpbdSolver: approaching velocity damped", "[solver][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp; sp.type = ShapeType::Sphere; sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 2);
    BodyParams p; p.mass = 1.f;
    // Body 0 at origin, Body 1 at (0,1.5,0) moving toward Body 0 (downward)
    p.position = {0.f, 0.f, 0.f}; p.linear_velocity = {0.f, 0.f, 0.f}; bs.add(p);
    p.position = {0.f, 1.5f, 0.f}; p.linear_velocity = {0.f, -2.f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    auto pbuf = one_pair(s, 0u, 1u);
    Narrowphase np(s, 8);
    np.run(s, pbuf.data(), 1u, bs.view(), ss.view());
    s.wait();

    REQUIRE(np.download_count(s) == 1u);

    XpbdSolver solver(s, 1);
    solver.solve(s, np.contacts(), JointView{}, bs.view(), 1.f/60.f);
    s.wait();

    auto& q = s.queue();
    // Normal is (0,-1,0) (from B toward A).  v_rel_n = (vy0 - vy1) * (-1).
    // After solve, vy0 and vy1 should be equal (conserved inelastic normal).
    float vy0 = download_f(q, bs.view().vel_y);
    float vy1 = download_f(q, bs.view().vel_y + 1);
    float vrel_n = (vy0 - vy1) * (-1.f);  // projected onto normal (0,-1,0)
    REQUIRE(vrel_n >= -kEps);  // no longer approaching
}

// Static body (mass=0) against a dynamic sphere. Only the dynamic body moves.
TEST_CASE("XpbdSolver: static body unmoved", "[solver][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp; sp.type = ShapeType::Sphere; sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 2);
    BodyParams p;
    // Body 0: static at origin
    p.mass = 0.f; p.flags = BodyFlag::Static;
    p.position = {0.f, 0.f, 0.f}; bs.add(p);
    // Body 1: dynamic at (0,1.5,0)
    p.mass = 1.f; p.flags = 0;
    p.position = {0.f, 1.5f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    auto pbuf = one_pair(s, 0u, 1u);
    Narrowphase np(s, 8);
    np.run(s, pbuf.data(), 1u, bs.view(), ss.view());
    s.wait();

    REQUIRE(np.download_count(s) == 1u);

    XpbdSolver solver(s, 1);
    solver.solve(s, np.contacts(), JointView{}, bs.view(), 1.f/60.f);
    s.wait();

    auto& q = s.queue();
    float y0 = download_f(q, bs.view().pos_y);
    float y1 = download_f(q, bs.view().pos_y + 1);

    REQUIRE_THAT(y0, WithinAbs(0.f, kEps));   // static body unmoved
    REQUIRE(y1 >= 1.f - kEps);                 // dynamic body pushed up
}

// Coloring correctness: the graph-colored parallel path must agree with the
// single-work-item serial oracle. We use N independent sphere pairs (no shared
// body) — enough of them to exceed kSerialThreshold so the colored path engages,
// but all assigned the same colour, so the two paths apply the *same* disjoint
// constraints and must produce bit-identical body state. (Cross-colour Gauss-
// Seidel ordering is exercised for determinism in sim_determinism_test.)
TEST_CASE("XpbdSolver: colored path == serial oracle (independent pairs)",
          "[solver]") {
    constexpr uint32_t N_PAIRS  = 300;                       // > kSerialThreshold
    constexpr uint32_t N_BODIES = N_PAIRS * 2;
    static_assert(N_PAIRS > XpbdSolver::kSerialThreshold);

    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    auto& q  = s.queue();

    ShapeStore ss(s, 1);
    ShapeParams sp; sp.type = ShapeType::Sphere; sp.half_x = 0.5f;
    ss.add(sp);
    ss.upload();

    // Each pair is widely separated from the others (Δy=10) so only the explicit
    // (2k, 2k+1) pair overlaps — no accidental cross-pair contacts.
    BodyStore bs(s, N_BODIES);
    std::vector<float> init_y(N_BODIES);
    BodyParams p; p.mass = 1.f; p.flags = 0;
    for (uint32_t k = 0; k < N_PAIRS; ++k) {
        float base = 10.f * static_cast<float>(k);
        p.position = {0.f, base,        0.f}; init_y[2*k]   = base;        bs.add(p);
        p.position = {0.f, base + 0.8f, 0.f}; init_y[2*k+1] = base + 0.8f; bs.add(p);
    }
    bs.upload(); s.wait();

    std::vector<ContactPair> pairs(N_PAIRS);
    for (uint32_t k = 0; k < N_PAIRS; ++k) pairs[k] = ContactPair{2*k, 2*k+1};
    Buffer<ContactPair> pbuf(s, N_PAIRS);
    pbuf.upload(pairs.data(), N_PAIRS);
    s.wait();

    Narrowphase np(s, N_PAIRS + 8);
    np.run(s, pbuf.data(), N_PAIRS, bs.view(), ss.view());
    s.wait();
    REQUIRE(np.download_count(s) == N_PAIRS);   // each pair overlaps once

    auto reset_bodies = [&]() {
        q.memcpy(bs.view().pos_y, init_y.data(), N_BODIES * sizeof(float)).wait();
    };
    auto snapshot_y = [&]() {
        std::vector<float> y(N_BODIES);
        q.memcpy(y.data(), bs.view().pos_y, N_BODIES * sizeof(float)).wait();
        return y;
    };

    // Colored path (default): n_con = 300 > threshold.
    reset_bodies();
    XpbdSolver colored(s, 10);
    colored.solve(s, np.contacts(), JointView{}, bs.view(), 1.f/60.f);
    s.wait();
    auto y_colored = snapshot_y();

    // Serial oracle on the identical initial state + identical contacts.
    reset_bodies();
    XpbdSolver serial(s, 10);
    serial.set_force_serial(true);
    serial.solve(s, np.contacts(), JointView{}, bs.view(), 1.f/60.f);
    s.wait();
    auto y_serial = snapshot_y();

    // Disjoint constraints ⇒ order-independent ⇒ bit-for-bit identical.
    for (uint32_t i = 0; i < N_BODIES; ++i)
        REQUIRE(y_colored[i] == y_serial[i]);

    // And the colored path is run-to-run deterministic.
    reset_bodies();
    colored.solve(s, np.contacts(), JointView{}, bs.view(), 1.f/60.f);
    s.wait();
    auto y_colored2 = snapshot_y();
    for (uint32_t i = 0; i < N_BODIES; ++i)
        REQUIRE(y_colored[i] == y_colored2[i]);
}
