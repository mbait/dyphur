#pragma once
#include "broadphase.hpp"
#include "body.hpp"
#include "shapes.hpp"
#include "contact_store.hpp"
#include "convex_hull_store.hpp"
#include "mesh_bvh.hpp"
#include <compute/buffer.hpp>
#include <compute/stream.hpp>
#include <cstdint>

namespace dyphur {

// Dispatches candidate pairs from the broadphase to primitive-vs-primitive
// contact tests and writes results to the internal ContactStore.
//
// Supported shape pairs (any ordering):
//   Sphere–Sphere     : exact distance test
//   Sphere–Box        : closest-point query on OBB
//   Box–Box           : SAT (15 axes) + vertex-face manifold (up to 4 contacts)
//   ConvexHull–*      : GJK distance + EPA penetration (against Box, Sphere, or Hull)
//   *–TriangleMesh    : BVH traversal + per-leaf Sphere/Box/ConvexHull–triangle test
//                       (TriangleMesh must be on a Static body)
//
// All submissions are async on s; call s.wait() before reading contacts.
class Narrowphase {
public:
    Narrowphase() = default;
    explicit Narrowphase(Stream& s, uint32_t max_contacts);

    // n_pairs must be downloaded from broadphase before calling.
    // hulls and meshes may be empty (n_hulls==0 / n_meshes==0) if no such shapes exist.
    void run(Stream& s,
             const ContactPair* d_pairs, uint32_t n_pairs,
             const BodyView& bodies, const ShapeView& shapes,
             ConvexHullView hulls = {},
             MeshBvhCatalogView meshes = {});

    // Async-capable variant: reads pair count from device memory (no CPU sync needed).
    // Submits all work to s without blocking; safe to chain sort→run→solve with a
    // single s.wait() or download_count at the frame boundary.
    void run(Stream& s,
             const ContactPair* d_pairs, const uint32_t* d_n_pairs,
             const BodyView& bodies, const ShapeView& shapes,
             ConvexHullView hulls = {},
             MeshBvhCatalogView meshes = {});

    ContactView contacts()   noexcept { return store_.view(); }
    uint32_t download_count(Stream& s) const { return store_.download_count(s); }

    // True number of contacts the most recent CPU-count run() generated, BEFORE
    // clamping to capacity. The narrowphase already syncs this count internally, so
    // reading it is free (no extra round-trip). If it exceeds the max_contacts the
    // narrowphase was constructed with, contacts were dropped in non-deterministic
    // (atomic-append) order and the result is NOT reproducible — size max_contacts
    // above this. Updated only by the CPU-pair-count run() overload.
    uint32_t last_contact_count() const noexcept { return last_count_; }
    bool     overflowed()         const noexcept { return last_count_ > cap_; }

private:
    // Pipeline: candidate pairs are tested in parallel (one work-item per pair),
    // appending contacts to scratch_ in non-deterministic order while tagging each
    // with a (pair_idx, sub_idx) key. The contacts are then sorted by that key and
    // gathered into store_, reproducing the exact order a sequential pass would
    // produce — so the solver/sensors and golden hashes are unaffected.
    ContactStore     store_;     // final, sorted, compact (read by solver/sensors)
    ContactStore     scratch_;   // parallel-append target (unordered)
    Buffer<uint64_t> keys_;      // per-contact sort key, padded to a power of two
    Buffer<uint32_t> perm_;      // identity → permutation that sorts keys_
    uint32_t         cap_        = 0;
    uint32_t         pad_cap_    = 0;  // next_pow2(cap_)
    uint32_t         last_count_ = 0;  // true contact count of last CPU-count run()
};

} // namespace dyphur
