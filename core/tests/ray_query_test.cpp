#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <core/aabb.hpp>
#include <core/body_store.hpp>
#include <core/broadphase.hpp>
#include <core/ray_query.hpp>
#include <core/shape_store.hpp>
#include <compute/buffer.hpp>
#include <compute/device.hpp>
#include <cstdint>
#include <vector>

using namespace dyphur;

static constexpr float kEps = 1e-3f;

// Helper: copy rays → device, launch ray_query, download hits → host.
static std::vector<RayHit> query(Stream& s,
                                 const Broadphase& bp, uint32_t n_bodies,
                                 const BodyView& bv, const ShapeView& sv,
                                 const std::vector<Ray>& rays) {
    auto& q = s.queue();
    uint32_t nr = static_cast<uint32_t>(rays.size());
    Ray*    d_rays = sycl::malloc_device<Ray>(nr, q);
    RayHit* d_hits = sycl::malloc_device<RayHit>(nr, q);

    q.memcpy(d_rays, rays.data(), nr * sizeof(Ray)).wait();

    BvhView bvh = bp.bvh_view(n_bodies);
    ray_query(s, bvh, bv, sv, d_rays, d_hits, nr);
    s.wait();

    std::vector<RayHit> out(nr);
    q.memcpy(out.data(), d_hits, nr * sizeof(RayHit)).wait();

    sycl::free(d_rays, q);
    sycl::free(d_hits, q);
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────

// Sphere at origin r=1; ray from (-5,0,0) along +x → t≈4 (enters at x=-1).
TEST_CASE("RayQuery: sphere hit along +x axis", "[ray_query][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp;
    sp.type   = ShapeType::Sphere;
    sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 4);
    BodyParams p;
    p.mass = 1.f;
    p.position = {0.f, 0.f, 0.f}; bs.add(p);
    p.position = {0.f, 50.f, 0.f}; bs.add(p);  // second body to force BVH build
    bs.upload();
    s.wait();

    AABB scene = {-6.f, -6.f, -6.f, 56.f, 56.f, 56.f};
    Broadphase bp(s, 4, 32);
    bp.build_and_query(s, bs.view(), ss.view(), scene);
    s.wait();

    std::vector<Ray> rays = {
        Ray{-5.f, 0.f, 0.f, 1.f, 0.f, 0.f},   // should hit body 0
        Ray{-5.f, 3.f, 0.f, 1.f, 0.f, 0.f},   // misses both
    };
    auto hits = query(s, bp, bs.view().n, bs.view(), ss.view(), rays);

    REQUIRE(hits[0].body_idx == 0u);
    REQUIRE_THAT(hits[0].t, Catch::Matchers::WithinAbs(4.f, kEps));

    REQUIRE(hits[1].body_idx == UINT32_MAX);
}

// Box at origin (1×1×1 half-extents); axis-aligned ray from (-5,0,0) along +x.
TEST_CASE("RayQuery: box hit along +x axis", "[ray_query][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp;
    sp.type   = ShapeType::Box;
    sp.half_x = sp.half_y = sp.half_z = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 4);
    BodyParams p;
    p.mass = 1.f;
    p.position = {0.f, 0.f, 0.f}; bs.add(p);
    p.position = {0.f, 50.f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    AABB scene = {-6.f, -6.f, -6.f, 56.f, 56.f, 56.f};
    Broadphase bp(s, 4, 32);
    bp.build_and_query(s, bs.view(), ss.view(), scene);
    s.wait();

    std::vector<Ray> rays = {
        Ray{-5.f, 0.f, 0.f, 1.f, 0.f, 0.f},   // hits face at x=-1 → t=4
        Ray{-5.f, 2.f, 0.f, 1.f, 0.f, 0.f},   // misses (y=2 > half_y=1)
    };
    auto hits = query(s, bp, bs.view().n, bs.view(), ss.view(), rays);

    REQUIRE(hits[0].body_idx == 0u);
    REQUIRE_THAT(hits[0].t, Catch::Matchers::WithinAbs(4.f, kEps));

    REQUIRE(hits[1].body_idx == UINT32_MAX);
}

// Two spheres along +x; ray should hit the nearer one (body 0 at x=0).
TEST_CASE("RayQuery: closest sphere returned when two overlap ray path", "[ray_query][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp;
    sp.type   = ShapeType::Sphere;
    sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 4);
    BodyParams p;
    p.mass = 1.f;
    p.position = { 0.f, 0.f, 0.f}; bs.add(p);  // body 0
    p.position = {10.f, 0.f, 0.f}; bs.add(p);  // body 1
    bs.upload();
    s.wait();

    AABB scene = {-6.f, -6.f, -6.f, 12.f, 6.f, 6.f};
    Broadphase bp(s, 4, 32);
    bp.build_and_query(s, bs.view(), ss.view(), scene);
    s.wait();

    std::vector<Ray> rays = {Ray{-5.f, 0.f, 0.f, 1.f, 0.f, 0.f}};
    auto hits = query(s, bp, bs.view().n, bs.view(), ss.view(), rays);

    // Must hit body 0 (closer), not body 1.
    REQUIRE(hits[0].body_idx == 0u);
    REQUIRE_THAT(hits[0].t, Catch::Matchers::WithinAbs(4.f, kEps));
}

// Ray with t_max set shorter than the sphere → miss.
TEST_CASE("RayQuery: t_max caps the hit distance", "[ray_query][smoke]") {
    auto dev = Device::default_cpu();
    auto s   = dev.make_stream();

    ShapeStore ss(s, 2);
    ShapeParams sp;
    sp.type   = ShapeType::Sphere;
    sp.half_x = 1.f;
    ss.add(sp);
    ss.upload();

    BodyStore bs(s, 4);
    BodyParams p;
    p.mass = 1.f;
    p.position = { 0.f, 0.f, 0.f}; bs.add(p);
    p.position = {50.f, 0.f, 0.f}; bs.add(p);
    bs.upload();
    s.wait();

    AABB scene = {-6.f, -6.f, -6.f, 56.f, 6.f, 6.f};
    Broadphase bp(s, 4, 32);
    bp.build_and_query(s, bs.view(), ss.view(), scene);
    s.wait();

    // Sphere at x=0 r=1; entry at t=4. Cap at t_max=2 → miss.
    Ray r{-5.f, 0.f, 0.f, 1.f, 0.f, 0.f};
    r.t_max = 2.f;
    auto hits = query(s, bp, bs.view().n, bs.view(), ss.view(), {r});

    REQUIRE(hits[0].body_idx == UINT32_MAX);
}
