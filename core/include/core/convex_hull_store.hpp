#pragma once
#include <compute/buffer.hpp>
#include <compute/stream.hpp>
#include <vector>
#include <cstdint>

namespace dyphur {

// Kernel-safe read-only view of all convex hull vertex arrays.
// Hulls are stored as a concatenated flat vertex array; per-hull start+count
// enable O(n_verts) brute-force support function (n_verts ≤ 64 for V-HACD).
struct ConvexHullView {
    const float*    vtx_x;
    const float*    vtx_y;
    const float*    vtx_z;
    const uint32_t* hull_start;  // hull i: vertices [hull_start[i], hull_start[i]+hull_count[i])
    const uint32_t* hull_count;
    uint32_t        n_hulls;
    uint32_t        n_verts;
};

// CPU-side hull catalog; add() one hull at a time, call upload() to push to GPU.
class ConvexHullStore {
public:
    ConvexHullStore() = default;
    explicit ConvexHullStore(Stream& s, uint32_t max_hulls, uint32_t max_total_verts);

    // Add a convex hull given its vertex positions; returns hull index.
    uint32_t add(const float* x, const float* y, const float* z, uint32_t n_verts);

    void           upload();
    ConvexHullView view() const noexcept;
    uint32_t       hull_count()  const noexcept { return n_hulls_; }
    uint32_t       total_verts() const noexcept { return n_verts_; }

private:
    Buffer<float>    d_vtx_x_, d_vtx_y_, d_vtx_z_;
    Buffer<uint32_t> d_hull_start_, d_hull_count_;

    std::vector<float>    h_vtx_x_, h_vtx_y_, h_vtx_z_;
    std::vector<uint32_t> h_hull_start_, h_hull_count_;

    uint32_t n_hulls_ = 0, n_verts_ = 0;
    uint32_t max_hulls_ = 0, max_verts_ = 0;
};

} // namespace dyphur
