#pragma once
#include "aabb.hpp"
#include "body.hpp"
#include "shapes.hpp"
#include <compute/buffer.hpp>
#include <compute/stream.hpp>
#include <cstdint>

namespace dyphur {

struct ContactPair {
    uint32_t a, b;  // body indices, a < b always
};

// LBVH broadphase (Karras 2012).
//
// Build flow per frame:
//   1. Morton codes from body centroids, sorted by key.
//   2. Karras binary radix tree: n leaves + (n-1) internal nodes.
//   3. Bottom-up AABB refit using per-node atomic flags.
//   4. Top-down traversal: each leaf queries the BVH; pairs (i<j) emitted once.
//
// Static / sleeping bodies are included in the BVH; type filtering is deferred
// to the narrowphase.
class Broadphase {
public:
    Broadphase() = default;

    // Allocates all GPU storage upfront.
    // max_bodies: upper bound on bodies.n passed to build_and_query.
    // max_pairs:  capacity of the output pair buffer; excess pairs are dropped.
    Broadphase(Stream& s, uint32_t max_bodies, uint32_t max_pairs);

    // Build LBVH and emit overlapping candidate pairs.
    // scene_bounds: AABB used to quantize body centroids to 10-bit Morton codes.
    // All submissions are async on s; call s.wait() before reading results.
    void build_and_query(Stream& s,
                         const BodyView&  bodies,
                         const ShapeView& shapes,
                         const AABB&      scene_bounds);

    // Device pointers to results — valid after s.wait().
    const ContactPair* pairs_ptr() const noexcept { return d_pairs_.data(); }
    const uint32_t*    count_ptr() const noexcept { return d_count_.data(); }
    uint32_t           max_pairs() const noexcept { return max_pairs_; }

    // Blocking download of the pair count.
    uint32_t download_count(Stream& s) const;

private:
    uint32_t max_bodies_ = 0;
    uint32_t max_padded_ = 0;  // next power of 2 >= max_bodies (for sort)
    uint32_t max_pairs_  = 0;

    // Bitonic sort input: one entry per body (padded to max_padded_).
    Buffer<uint32_t> d_morton_;
    Buffer<uint32_t> d_sorted_idx_;  // sorted_idx[k] = original body index of leaf k

    // BVH node arrays — total 2*max_bodies-1 nodes:
    //   internal nodes: [0, max_bodies-2]
    //   leaf nodes:     [max_bodies-1, 2*max_bodies-2]  (leaf k → node max_bodies-1+k)
    Buffer<int32_t>  d_left_;      // size max_bodies-1: left child index of internal node i
    Buffer<int32_t>  d_right_;     // size max_bodies-1: right child index
    Buffer<int32_t>  d_parent_;    // size 2*max_bodies-1: parent index (-1 for root)
    Buffer<float>    d_aabb_min_x_, d_aabb_min_y_, d_aabb_min_z_;
    Buffer<float>    d_aabb_max_x_, d_aabb_max_y_, d_aabb_max_z_;
    Buffer<uint32_t> d_flags_;     // size max_bodies-1: atomic refit counters
    Buffer<int32_t>  d_root_;      // 1 element: index of the root internal node

    Buffer<ContactPair> d_pairs_;
    Buffer<uint32_t>    d_count_;  // 1 element: emitted pair count
};

} // namespace dyphur
