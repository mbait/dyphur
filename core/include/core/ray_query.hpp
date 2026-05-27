#pragma once
#include "broadphase.hpp"
#include "body.hpp"
#include "shapes.hpp"
#include <compute/stream.hpp>
#include <cstdint>

namespace dyphur {

struct Ray {
    float ox, oy, oz;   // origin
    float dx, dy, dz;   // direction (need not be normalised; t is in direction units)
    float t_max = 1e9f; // maximum distance; hit only recorded if t < t_max
};

struct RayHit {
    float    t;          // distance along ray (in direction units)
    uint32_t body_idx;   // UINT32_MAX when no hit
};

// Batched BVH ray query.
//
// Traverses the LBVH built by Broadphase::build_and_query() for n_rays rays in
// parallel.  Each ray returns the closest hit (smallest t) against any body's
// world-space AABB at the leaf level, with an exact per-shape intersection test:
//   Box:    OBB slab test in body-local frame.
//   Sphere: analytic quadratic.
//
// d_rays and d_hits must be device-accessible (USM or device Buffer data pointers).
// All submissions are async on s; call s.wait() before reading d_hits.
//
// Precondition: bvh must have been populated by build_and_query() + s.wait()
// and bvh.n must equal the body count used for that call.
void ray_query(Stream& s,
               const BvhView&  bvh,
               const BodyView& bodies,
               const ShapeView& shapes,
               const Ray*       d_rays,
               RayHit*          d_hits,
               uint32_t         n_rays);

} // namespace dyphur
