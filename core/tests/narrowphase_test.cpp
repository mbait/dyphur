#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <core/body_store.hpp>
#include <core/narrowphase.hpp>
#include <core/shape_store.hpp>
#include <compute/buffer.hpp>
#include <compute/device.hpp>
#include <vector>

using namespace dyphur;
using Catch::Matchers::WithinAbs;

static constexpr float kEps = 1e-4f;

struct Contacts {
    std::vector<uint32_t> body_a, body_b;
    std::vector<float> px, py, pz;
    std::vector<float> nx, ny, nz;
    std::vector<float> depth;
    uint32_t n = 0;
};

static Contacts download_contacts(Stream& s, Narrowphase& np) {
    uint32_t n = np.download_count(s);
    Contacts c; c.n = n;
    if (n == 0) return c;
    ContactView cv = np.contacts();
    auto& q = s.queue();
    c.body_a.resize(n); c.body_b.resize(n);
    c.px.resize(n); c.py.resize(n); c.pz.resize(n);
    c.nx.resize(n); c.ny.resize(n); c.nz.resize(n);
    c.depth.resize(n);
    q.memcpy(c.body_a.data(), cv.body_a, n * sizeof(uint32_t));
    q.memcpy(c.body_b.data(), cv.body_b, n * sizeof(uint32_t));
    q.memcpy(c.px.data(), cv.pos_x,  n * sizeof(float));
    q.memcpy(c.py.data(), cv.pos_y,  n * sizeof(float));
    q.memcpy(c.pz.data(), cv.pos_z,  n * sizeof(float));
    q.memcpy(c.nx.data(), cv.norm_x, n * sizeof(float));
    q.memcpy(c.ny.data(), cv.norm_y, n * sizeof(float));
    q.memcpy(c.nz.data(), cv.norm_z, n * sizeof(float));
    q.memcpy(c.depth.data(), cv.depth, n * sizeof(float)).wait();
    return c;
}

static Buffer<ContactPair> make_pairs(Stream& s,
                                      std::initializer_list<ContactPair> pairs) {
    Buffer<ContactPair> buf(s, pairs.size());
    std::vector<ContactPair> h(pairs);
    buf.upload(h.data(), h.size());
    s.wait();
    return buf;
}

// ─────────────────────────────────────────────────────────────────────────────

// Sphere A radius 1 at (0,0,0), sphere B radius 1 at (1.5,0,0).
// dist=1.5, depth=0.5, N=(-1,0,0), contact midpoint=(0.75,0,0).
TEST_CASE("Narrowphase: sphere-sphere contact", "[narrowphase][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp; sp.type = ShapeType::Sphere; sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 2);
    BodyParams p; p.mass = 1.f;
    p.position = {0.f,  0.f, 0.f}; bs.add(p);
    p.position = {1.5f, 0.f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    auto pbuf = make_pairs(s, {{0u, 1u}});
    Narrowphase np(s, 8);
    np.run(s, pbuf.data(), 1u, bs.view(), ss.view());
    s.wait();

    auto c = download_contacts(s, np);
    REQUIRE(c.n == 1u);
    REQUIRE_THAT(c.depth[0], WithinAbs(0.5f,  kEps));
    REQUIRE_THAT(c.nx[0],    WithinAbs(-1.f,  kEps));
    REQUIRE_THAT(c.ny[0],    WithinAbs(0.f,   kEps));
    REQUIRE_THAT(c.nz[0],    WithinAbs(0.f,   kEps));
    REQUIRE_THAT(c.px[0],    WithinAbs(0.75f, kEps));
    REQUIRE_THAT(c.py[0],    WithinAbs(0.f,   kEps));
    REQUIRE_THAT(c.pz[0],    WithinAbs(0.f,   kEps));
}

// dist=3 > 2, no contact.
TEST_CASE("Narrowphase: sphere-sphere no contact", "[narrowphase][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp; sp.type = ShapeType::Sphere; sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 2);
    BodyParams p; p.mass = 1.f;
    p.position = {0.f, 0.f, 0.f}; bs.add(p);
    p.position = {3.f, 0.f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    auto pbuf = make_pairs(s, {{0u, 1u}});
    Narrowphase np(s, 8);
    np.run(s, pbuf.data(), 1u, bs.view(), ss.view());
    s.wait();

    REQUIRE(np.download_count(s) == 0u);
}

// body 0: box 1×1×1 at (0,0,0); body 1: sphere r=0.5 at (1.3,0,0).
// Closest box pt=(1,0,0), depth=0.2, N from B(sphere)->A(box)=(-1,0,0).
TEST_CASE("Narrowphase: sphere-box contact", "[narrowphase][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 4);
    ShapeParams box_sp; box_sp.type = ShapeType::Box;
    box_sp.half_x = 1.f; box_sp.half_y = 1.f; box_sp.half_z = 1.f;
    ss.add(box_sp);  // shape 0 = box

    ShapeParams sph_sp; sph_sp.type = ShapeType::Sphere; sph_sp.half_x = 0.5f;
    ss.add(sph_sp);  // shape 1 = sphere
    ss.upload();

    BodyStore bs(s, 4);
    BodyParams p; p.mass = 1.f;
    p.shape_handle = 0; p.position = {0.f,  0.f, 0.f}; bs.add(p);
    p.shape_handle = 1; p.position = {1.3f, 0.f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    auto pbuf = make_pairs(s, {{0u, 1u}});
    Narrowphase np(s, 8);
    np.run(s, pbuf.data(), 1u, bs.view(), ss.view());
    s.wait();

    auto c = download_contacts(s, np);
    REQUIRE(c.n == 1u);
    REQUIRE_THAT(c.depth[0], WithinAbs(0.2f, kEps));
    REQUIRE_THAT(c.nx[0],    WithinAbs(-1.f, kEps));
    REQUIRE_THAT(c.ny[0],    WithinAbs(0.f,  kEps));
    REQUIRE_THAT(c.nz[0],    WithinAbs(0.f,  kEps));
    REQUIRE_THAT(c.px[0],    WithinAbs(1.f,  kEps));
    REQUIRE_THAT(c.py[0],    WithinAbs(0.f,  kEps));
    REQUIRE_THAT(c.pz[0],    WithinAbs(0.f,  kEps));
}

// Two 1×1×1 boxes: A at (0,0,0), B at (1.8,0,0). Overlap=0.2.
// Expect 1 centroid contact at x=1.0, depth=0.2, N=(-1,0,0).
TEST_CASE("Narrowphase: box-box face contact", "[narrowphase][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams box_sp; box_sp.type = ShapeType::Box;
    box_sp.half_x = 1.f; box_sp.half_y = 1.f; box_sp.half_z = 1.f;
    ss.add(box_sp);
    ss.upload();

    BodyStore bs(s, 2);
    BodyParams p; p.mass = 1.f;
    p.position = {0.f,  0.f, 0.f}; bs.add(p);
    p.position = {1.8f, 0.f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    auto pbuf = make_pairs(s, {{0u, 1u}});
    Narrowphase np(s, 16);
    np.run(s, pbuf.data(), 1u, bs.view(), ss.view());
    s.wait();

    auto c = download_contacts(s, np);
    REQUIRE(c.n == 1u);
    REQUIRE_THAT(c.depth[0], WithinAbs(0.2f, kEps));
    REQUIRE_THAT(c.nx[0],    WithinAbs(-1.f, kEps));
    REQUIRE_THAT(c.ny[0],    WithinAbs(0.f,  kEps));
    REQUIRE_THAT(c.nz[0],    WithinAbs(0.f,  kEps));
    REQUIRE_THAT(c.px[0],    WithinAbs(1.f,  kEps));
}

// Two 1×1×1 boxes 5 units apart — no contact.
TEST_CASE("Narrowphase: box-box no overlap", "[narrowphase][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams box_sp; box_sp.type = ShapeType::Box;
    box_sp.half_x = 1.f; box_sp.half_y = 1.f; box_sp.half_z = 1.f;
    ss.add(box_sp);
    ss.upload();

    BodyStore bs(s, 2);
    BodyParams p; p.mass = 1.f;
    p.position = {0.f, 0.f, 0.f}; bs.add(p);
    p.position = {5.f, 0.f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    auto pbuf = make_pairs(s, {{0u, 1u}});
    Narrowphase np(s, 8);
    np.run(s, pbuf.data(), 1u, bs.view(), ss.view());
    s.wait();

    REQUIRE(np.download_count(s) == 0u);
}
