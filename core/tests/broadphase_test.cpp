#include <catch2/catch_test_macros.hpp>
#include <core/aabb.hpp>
#include <core/body_store.hpp>
#include <core/broadphase.hpp>
#include <core/shape_store.hpp>
#include <compute/device.hpp>
#include <algorithm>
#include <vector>

using namespace dyphur;

static std::vector<ContactPair> download_pairs(sycl::queue& q,
                                               const Broadphase& bp,
                                               uint32_t n) {
    std::vector<ContactPair> out(n);
    if (n > 0)
        q.memcpy(out.data(), bp.pairs_ptr(), n * sizeof(ContactPair)).wait();
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────

TEST_CASE("AABB: overlap and non-overlap", "[broadphase][aabb][smoke]") {
    AABB a = { 0.f, 0.f, 0.f,  1.f, 1.f, 1.f };
    AABB b = { 0.5f, 0.f, 0.f, 1.5f, 1.f, 1.f };  // overlaps a in X
    AABB c = { 2.f, 0.f, 0.f,  3.f, 1.f, 1.f };   // disjoint from a

    REQUIRE(a.overlaps(b));
    REQUIRE(b.overlaps(a));
    REQUIRE_FALSE(a.overlaps(c));
    REQUIRE_FALSE(c.overlaps(a));

    // touching edge — counts as overlap (min_x == max_x boundary)
    AABB d = { 1.f, 0.f, 0.f, 2.f, 1.f, 1.f };
    REQUIRE(a.overlaps(d));
}

TEST_CASE("AABB: merge", "[broadphase][aabb][smoke]") {
    AABB a = { 0.f, 0.f, 0.f, 1.f, 1.f, 1.f };
    AABB b = { 0.5f, -1.f, 0.f, 2.f, 0.5f, 3.f };
    AABB m = AABB::merge(a, b);
    REQUIRE(m.min_x == 0.f);   REQUIRE(m.max_x == 2.f);
    REQUIRE(m.min_y == -1.f);  REQUIRE(m.max_y == 1.f);
    REQUIRE(m.min_z == 0.f);   REQUIRE(m.max_z == 3.f);
}

// ─────────────────────────────────────────────────────────────────────────────

// 3 spheres of radius 1:
//   0 at (0,0,0), 1 at (1,0,0) → overlap (dist=1 < r0+r1=2)
//   2 at (100,0,0)              → disjoint from both
// Expected: exactly 1 pair = {0, 1}.
TEST_CASE("Broadphase: three spheres, one overlapping pair", "[broadphase][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();
    auto& q  = s.queue();

    ShapeStore ss(s, 4);
    ShapeParams sp;
    sp.type   = ShapeType::Sphere;
    sp.half_x = 1.f;  // radius
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 4);
    BodyParams p;
    p.mass = 1.f;

    p.position = {  0.f, 0.f, 0.f }; bs.add(p);
    p.position = {  1.f, 0.f, 0.f }; bs.add(p);
    p.position = {100.f, 0.f, 0.f }; bs.add(p);
    bs.upload();
    s.wait();

    Broadphase bp(s, 4, 32);
    AABB scene = { -2.f, -2.f, -2.f, 102.f, 2.f, 2.f };
    bp.build_and_query(s, bs.view(), ss.view(), scene);
    s.wait();

    uint32_t cnt = bp.download_count(s);
    REQUIRE(cnt == 1u);

    auto pairs = download_pairs(q, bp, cnt);
    // Pair must be (0,1) with a < b enforced.
    REQUIRE(pairs[0].a == 0u);
    REQUIRE(pairs[0].b == 1u);
}

// All 4 bodies overlap their neighbours; expect all 6 pairs.
TEST_CASE("Broadphase: four overlapping spheres, all pairs", "[broadphase][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 4);
    ShapeParams sp;
    sp.type   = ShapeType::Sphere;
    sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 4);
    BodyParams p;
    p.mass = 1.f;
    p.position = { 0.f, 0.f, 0.f }; bs.add(p);
    p.position = { 0.5f, 0.f, 0.f }; bs.add(p);
    p.position = { 0.f, 0.5f, 0.f }; bs.add(p);
    p.position = { 0.f, 0.f, 0.5f }; bs.add(p);
    bs.upload();
    s.wait();

    Broadphase bp(s, 4, 32);
    AABB scene = { -2.f, -2.f, -2.f, 2.f, 2.f, 2.f };
    bp.build_and_query(s, bs.view(), ss.view(), scene);
    s.wait();

    uint32_t cnt = bp.download_count(s);
    REQUIRE(cnt == 6u);

    auto pairs = download_pairs(s.queue(), bp, cnt);
    // All pairs must have a < b.
    for (auto& cp : pairs)
        REQUIRE(cp.a < cp.b);
    // Each pair is unique.
    std::sort(pairs.begin(), pairs.end(), [](const ContactPair& x, const ContactPair& y) {
        return x.a < y.a || (x.a == y.a && x.b < y.b);
    });
    for (size_t i = 1; i < pairs.size(); ++i) {
        bool dup = (pairs[i].a == pairs[i-1].a) && (pairs[i].b == pairs[i-1].b);
        REQUIRE_FALSE(dup);
    }
}

// Two isolated spheres that do not overlap → zero pairs.
TEST_CASE("Broadphase: two non-overlapping spheres", "[broadphase][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp;
    sp.type   = ShapeType::Sphere;
    sp.half_x = 0.4f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 2);
    BodyParams p;
    p.mass = 1.f;
    p.position = { 0.f, 0.f, 0.f }; bs.add(p);
    p.position = { 5.f, 0.f, 0.f }; bs.add(p);
    bs.upload();
    s.wait();

    Broadphase bp(s, 4, 32);
    AABB scene = { -1.f, -1.f, -1.f, 6.f, 1.f, 1.f };
    bp.build_and_query(s, bs.view(), ss.view(), scene);
    s.wait();

    REQUIRE(bp.download_count(s) == 0u);
}
