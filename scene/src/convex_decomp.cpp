// V-HACD is a header-only library; define implementation exactly once here.
#define ENABLE_VHACD_IMPLEMENTATION 1
#include <VHACD.h>

#include <scene/convex_decomp.hpp>
#include <stdexcept>

namespace dyphur {

std::vector<VertexBuffer> decompose_vhacd(const VertexBuffer& mesh, const DecompParams& p)
{
    if (mesh.x.empty() || mesh.idx_a.empty())
        throw std::runtime_error("decompose_vhacd: mesh is empty");

    VHACD::IVHACD::Parameters vp;
    vp.m_maxConvexHulls    = p.max_hulls;
    vp.m_resolution        = p.resolution;
    vp.m_maxNumVerticesPerCH = p.max_verts_per_hull;
    vp.m_minimumVolumePercentErrorAllowed = p.min_volume_percent_error;

    VHACD::IVHACD* vhacd = VHACD::CreateVHACD();

    // Flat float vertex array (X, Y, Z interleaved)
    std::vector<float> points;
    points.reserve(mesh.x.size() * 3);
    for (size_t i = 0; i < mesh.x.size(); ++i) {
        points.push_back(mesh.x[i]);
        points.push_back(mesh.y[i]);
        points.push_back(mesh.z[i]);
    }
    // Flat uint triangle array (A, B, C interleaved)
    std::vector<uint32_t> triangles;
    triangles.reserve(mesh.idx_a.size() * 3);
    for (size_t i = 0; i < mesh.idx_a.size(); ++i) {
        triangles.push_back(mesh.idx_a[i]);
        triangles.push_back(mesh.idx_b[i]);
        triangles.push_back(mesh.idx_c[i]);
    }

    bool ok = vhacd->Compute(points.data(),
                              static_cast<uint32_t>(mesh.x.size()),
                              triangles.data(),
                              static_cast<uint32_t>(mesh.idx_a.size()),
                              vp);
    if (!ok) {
        vhacd->Release();
        throw std::runtime_error("V-HACD decomposition failed");
    }

    std::vector<VertexBuffer> result;
    uint32_t n_hulls = vhacd->GetNConvexHulls();
    result.reserve(n_hulls);

    for (uint32_t hi = 0; hi < n_hulls; ++hi) {
        VHACD::IVHACD::ConvexHull hull;
        vhacd->GetConvexHull(hi, hull);

        VertexBuffer vb;
        vb.x.reserve(hull.m_points.size());
        vb.y.reserve(hull.m_points.size());
        vb.z.reserve(hull.m_points.size());
        for (const auto& pt : hull.m_points) {
            vb.x.push_back(static_cast<float>(pt.mX));
            vb.y.push_back(static_cast<float>(pt.mY));
            vb.z.push_back(static_cast<float>(pt.mZ));
        }
        result.push_back(std::move(vb));
    }

    vhacd->Release();
    return result;
}

} // namespace dyphur
