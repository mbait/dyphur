#pragma once
#include "broadphase.hpp"
#include "body.hpp"
#include "shapes.hpp"
#include "contact_store.hpp"
#include "convex_hull_store.hpp"
#include "mesh_bvh.hpp"
#include <compute/stream.hpp>

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

private:
    ContactStore store_;
};

} // namespace dyphur
