#include <core/mesh_bvh.hpp>
#include <algorithm>
#include <cassert>
#include <limits>
#include <numeric>

namespace dyphur {

// ── MeshBvh CPU builder ───────────────────────────────────────────────────────

void MeshBvh::build(const float* vx, const float* vy, const float* vz, uint32_t n_verts,
                    const uint32_t* ta, const uint32_t* tb, const uint32_t* tc, uint32_t n_tris)
{
    h_vtx_x_.assign(vx, vx + n_verts);
    h_vtx_y_.assign(vy, vy + n_verts);
    h_vtx_z_.assign(vz, vz + n_verts);
    h_tri_a_.assign(ta, ta + n_tris);
    h_tri_b_.assign(tb, tb + n_tris);
    h_tri_c_.assign(tc, tc + n_tris);
    h_nodes_.clear();

    std::vector<uint32_t> order(n_tris);
    std::iota(order.begin(), order.end(), 0);
    root_ = static_cast<uint32_t>(build_node(order, 0, static_cast<int>(n_tris)));
}

int MeshBvh::build_node(std::vector<uint32_t>& tris, int lo, int hi)
{
    assert(lo < hi);

    // Compute AABB over [lo, hi)
    float min_x = std::numeric_limits<float>::max();
    float min_y = min_x, min_z = min_x;
    float max_x = -min_x, max_y = -min_x, max_z = -min_x;

    for (int i = lo; i < hi; ++i) {
        uint32_t t = tris[i];
        for (uint32_t v : {h_tri_a_[t], h_tri_b_[t], h_tri_c_[t]}) {
            min_x = std::min(min_x, h_vtx_x_[v]);
            min_y = std::min(min_y, h_vtx_y_[v]);
            min_z = std::min(min_z, h_vtx_z_[v]);
            max_x = std::max(max_x, h_vtx_x_[v]);
            max_y = std::max(max_y, h_vtx_y_[v]);
            max_z = std::max(max_z, h_vtx_z_[v]);
        }
    }

    int node_idx = static_cast<int>(h_nodes_.size());
    h_nodes_.push_back({});
    BvhNode& node = h_nodes_.back();
    node.min_x = min_x; node.min_y = min_y; node.min_z = min_z;
    node.max_x = max_x; node.max_y = max_y; node.max_z = max_z;
    node._pad = 0;

    if (hi - lo == 1) {
        node.left = node.right = -1;
        node.tri_idx = static_cast<int32_t>(tris[lo]);
        return node_idx;
    }

    // Longest-axis median split
    float ex = max_x - min_x, ey = max_y - min_y, ez = max_z - min_z;
    int axis = (ex >= ey && ex >= ez) ? 0 : (ey >= ez ? 1 : 2);

    auto centroid = [&](uint32_t t) -> float {
        uint32_t a = h_tri_a_[t], b = h_tri_b_[t], c = h_tri_c_[t];
        if (axis == 0) return (h_vtx_x_[a] + h_vtx_x_[b] + h_vtx_x_[c]) / 3.f;
        if (axis == 1) return (h_vtx_y_[a] + h_vtx_y_[b] + h_vtx_y_[c]) / 3.f;
        return (h_vtx_z_[a] + h_vtx_z_[b] + h_vtx_z_[c]) / 3.f;
    };

    int mid = (lo + hi) / 2;
    std::nth_element(tris.begin() + lo, tris.begin() + mid, tris.begin() + hi,
                     [&](uint32_t a, uint32_t b) { return centroid(a) < centroid(b); });

    node.tri_idx = -1;
    int left  = build_node(tris, lo, mid);
    // Re-fetch reference since h_nodes_ may have reallocated
    int right = build_node(tris, mid, hi);
    h_nodes_[node_idx].left  = left;
    h_nodes_[node_idx].right = right;
    return node_idx;
}

// ── MeshBvhStore ─────────────────────────────────────────────────────────────

MeshBvhStore::MeshBvhStore(Stream& s,
                            uint32_t max_meshes,
                            uint32_t max_total_verts,
                            uint32_t max_total_tris,
                            uint32_t max_total_nodes)
    : d_vtx_offset_(s, max_meshes), d_tri_offset_(s, max_meshes)
    , d_node_offset_(s, max_meshes), d_root_node_(s, max_meshes)
    , d_vtx_x_(s, max_total_verts), d_vtx_y_(s, max_total_verts), d_vtx_z_(s, max_total_verts)
    , d_tri_a_(s, max_total_tris), d_tri_b_(s, max_total_tris), d_tri_c_(s, max_total_tris)
    , d_nodes_(s, max_total_nodes)
{
    h_vtx_offset_.reserve(max_meshes);
    h_tri_offset_.reserve(max_meshes);
    h_node_offset_.reserve(max_meshes);
    h_root_node_.reserve(max_meshes);
    h_vtx_x_.reserve(max_total_verts);
    h_vtx_y_.reserve(max_total_verts);
    h_vtx_z_.reserve(max_total_verts);
    h_tri_a_.reserve(max_total_tris);
    h_tri_b_.reserve(max_total_tris);
    h_tri_c_.reserve(max_total_tris);
    h_nodes_.reserve(max_total_nodes);
}

uint32_t MeshBvhStore::add(const MeshBvh& bvh)
{
    auto vtx_off  = static_cast<uint32_t>(h_vtx_x_.size());
    auto tri_off  = static_cast<uint32_t>(h_tri_a_.size());
    auto node_off = static_cast<uint32_t>(h_nodes_.size());

    h_vtx_offset_.push_back(vtx_off);
    h_tri_offset_.push_back(tri_off);
    h_node_offset_.push_back(node_off);
    h_root_node_.push_back(node_off + bvh.root());

    h_vtx_x_.insert(h_vtx_x_.end(), bvh.vtx_x().begin(), bvh.vtx_x().end());
    h_vtx_y_.insert(h_vtx_y_.end(), bvh.vtx_y().begin(), bvh.vtx_y().end());
    h_vtx_z_.insert(h_vtx_z_.end(), bvh.vtx_z().begin(), bvh.vtx_z().end());
    h_tri_a_.insert(h_tri_a_.end(), bvh.tri_a().begin(), bvh.tri_a().end());
    h_tri_b_.insert(h_tri_b_.end(), bvh.tri_b().begin(), bvh.tri_b().end());
    h_tri_c_.insert(h_tri_c_.end(), bvh.tri_c().begin(), bvh.tri_c().end());

    // Offset node child indices by node_off so they point into the shared node array
    for (const BvhNode& nd : bvh.nodes()) {
        BvhNode n = nd;
        if (n.left  >= 0) n.left  += static_cast<int32_t>(node_off);
        if (n.right >= 0) n.right += static_cast<int32_t>(node_off);
        h_nodes_.push_back(n);
    }

    return n_meshes_++;
}

void MeshBvhStore::upload()
{
    if (n_meshes_ == 0) return;
    uint32_t nv = static_cast<uint32_t>(h_vtx_x_.size());
    uint32_t nt = static_cast<uint32_t>(h_tri_a_.size());
    uint32_t nn = static_cast<uint32_t>(h_nodes_.size());
    d_vtx_offset_.upload(h_vtx_offset_.data(), n_meshes_);
    d_tri_offset_.upload(h_tri_offset_.data(), n_meshes_);
    d_node_offset_.upload(h_node_offset_.data(), n_meshes_);
    d_root_node_.upload(h_root_node_.data(), n_meshes_);
    d_vtx_x_.upload(h_vtx_x_.data(), nv);
    d_vtx_y_.upload(h_vtx_y_.data(), nv);
    d_vtx_z_.upload(h_vtx_z_.data(), nv);
    d_tri_a_.upload(h_tri_a_.data(), nt);
    d_tri_b_.upload(h_tri_b_.data(), nt);
    d_tri_c_.upload(h_tri_c_.data(), nt);
    d_nodes_.upload(h_nodes_.data(), nn);
}

MeshBvhCatalogView MeshBvhStore::view() const noexcept
{
    return {
        d_vtx_offset_.data(), d_tri_offset_.data(),
        d_node_offset_.data(), d_root_node_.data(),
        d_vtx_x_.data(), d_vtx_y_.data(), d_vtx_z_.data(),
        d_tri_a_.data(), d_tri_b_.data(), d_tri_c_.data(),
        d_nodes_.data(),
        n_meshes_
    };
}

} // namespace dyphur
