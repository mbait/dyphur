#pragma once
#include <compute/buffer.hpp>
#include <compute/stream.hpp>
#include <cstdint>
#include <vector>

namespace dyphur {

// Flat BVH node for GPU traversal.
// Leaves have left == right == -1 and a valid tri_idx.
// Internal nodes have tri_idx == -1.
struct BvhNode {
    float    min_x, min_y, min_z;
    float    max_x, max_y, max_z;
    int32_t  left;     // child index; -1 for leaf
    int32_t  right;    // child index; -1 for leaf
    int32_t  tri_idx;  // valid when leaf
    int32_t  _pad;
};

// GPU read-only view of a single mesh BVH.
struct MeshBvhView {
    const float*    vtx_x;
    const float*    vtx_y;
    const float*    vtx_z;
    const uint32_t* tri_a;
    const uint32_t* tri_b;
    const uint32_t* tri_c;
    const BvhNode*  nodes;
    uint32_t        root;
    uint32_t        n_nodes;
    uint32_t        n_tris;
    uint32_t        n_verts;
};

// GPU read-only catalog of many mesh BVHs (one per TriangleMesh shape).
// All data is stored in flat shared arrays; per-mesh offsets index into them.
struct MeshBvhCatalogView {
    // Per-mesh offsets (indexed by shape ext_id)
    const uint32_t* vtx_offset;   // start of vertices for mesh i
    const uint32_t* tri_offset;   // start of triangles for mesh i
    const uint32_t* node_offset;  // start of BVH nodes for mesh i
    const uint32_t* root_node;    // root node index (absolute) for mesh i

    // Shared flat arrays
    const float*    vtx_x;
    const float*    vtx_y;
    const float*    vtx_z;
    const uint32_t* tri_a;
    const uint32_t* tri_b;
    const uint32_t* tri_c;
    const BvhNode*  nodes;

    uint32_t n_meshes;
};

// CPU-side builder and GPU store for a single mesh BVH.
class MeshBvh {
public:
    MeshBvh() = default;

    // Build BVH from triangle soup; all data is in local (body) space.
    void build(const float* vx, const float* vy, const float* vz, uint32_t n_verts,
               const uint32_t* ta, const uint32_t* tb, const uint32_t* tc, uint32_t n_tris);

    const std::vector<float>&    vtx_x() const { return h_vtx_x_; }
    const std::vector<float>&    vtx_y() const { return h_vtx_y_; }
    const std::vector<float>&    vtx_z() const { return h_vtx_z_; }
    const std::vector<uint32_t>& tri_a() const { return h_tri_a_; }
    const std::vector<uint32_t>& tri_b() const { return h_tri_b_; }
    const std::vector<uint32_t>& tri_c() const { return h_tri_c_; }
    const std::vector<BvhNode>&  nodes() const { return h_nodes_; }
    uint32_t root()    const { return root_; }
    uint32_t n_tris()  const { return static_cast<uint32_t>(h_tri_a_.size()); }
    uint32_t n_verts() const { return static_cast<uint32_t>(h_vtx_x_.size()); }

private:
    std::vector<float>    h_vtx_x_, h_vtx_y_, h_vtx_z_;
    std::vector<uint32_t> h_tri_a_, h_tri_b_, h_tri_c_;
    std::vector<BvhNode>  h_nodes_;
    uint32_t root_ = 0;

    int build_node(std::vector<uint32_t>& tris, int lo, int hi);
};

// Stores multiple MeshBvh objects, uploads them to GPU, and provides a catalog view.
class MeshBvhStore {
public:
    MeshBvhStore() = default;
    explicit MeshBvhStore(Stream& s,
                          uint32_t max_meshes,
                          uint32_t max_total_verts,
                          uint32_t max_total_tris,
                          uint32_t max_total_nodes);

    // Add a pre-built MeshBvh; returns mesh index (used as shape ext_id).
    uint32_t add(const MeshBvh& bvh);

    void               upload();
    MeshBvhCatalogView view() const noexcept;
    uint32_t           mesh_count() const noexcept { return n_meshes_; }

private:
    Buffer<uint32_t> d_vtx_offset_, d_tri_offset_, d_node_offset_, d_root_node_;
    Buffer<float>    d_vtx_x_, d_vtx_y_, d_vtx_z_;
    Buffer<uint32_t> d_tri_a_, d_tri_b_, d_tri_c_;
    Buffer<BvhNode>  d_nodes_;

    std::vector<uint32_t> h_vtx_offset_, h_tri_offset_, h_node_offset_, h_root_node_;
    std::vector<float>    h_vtx_x_, h_vtx_y_, h_vtx_z_;
    std::vector<uint32_t> h_tri_a_, h_tri_b_, h_tri_c_;
    std::vector<BvhNode>  h_nodes_;

    uint32_t n_meshes_ = 0;
};

} // namespace dyphur
