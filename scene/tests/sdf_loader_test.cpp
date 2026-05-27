#include <catch2/catch_test_macros.hpp>
#include <scene/scene_desc.hpp>

#ifdef DYPHUR_SCENE_ENABLED
#include <scene/sdf_loader.hpp>
#include <scene/convex_decomp.hpp>
#endif

using namespace dyphur;

TEST_CASE("scene sdf_loader module links", "[smoke]") {
    REQUIRE(true);
}

#ifdef DYPHUR_SCENE_ENABLED

TEST_CASE("VertexBuffer round-trip (no SDF required)", "[smoke]") {
    // Tetrahedron
    VertexBuffer vb;
    vb.x = {0,1,0,0}; vb.y = {0,0,1,0}; vb.z = {0,0,0,1};
    vb.idx_a = {0,0,0,1}; vb.idx_b = {1,2,3,2}; vb.idx_c = {2,3,1,3};
    REQUIRE(vb.x.size() == 4);
    REQUIRE(vb.idx_a.size() == 4);
}

TEST_CASE("V-HACD convex decomp: tetrahedron → ≥1 hull", "[smoke]") {
    VertexBuffer vb;
    vb.x = {0,1,0,0}; vb.y = {0,0,1,0}; vb.z = {0,0,0,1};
    vb.idx_a = {0,0,0,1}; vb.idx_b = {1,2,3,2}; vb.idx_c = {2,3,1,3};

    DecompParams dp; dp.max_hulls = 4;
    auto hulls = decompose_vhacd(vb, dp);
    REQUIRE(!hulls.empty());
    for (const auto& h : hulls) {
        REQUIRE(!h.x.empty());
        REQUIRE(h.x.size() == h.y.size());
        REQUIRE(h.x.size() == h.z.size());
    }
}

#endif // DYPHUR_SCENE_ENABLED
