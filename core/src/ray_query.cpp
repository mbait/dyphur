#include <core/ray_query.hpp>
#include <core/math/math.hpp>
#include <compute/kernel.hpp>
#include <sycl/sycl.hpp>
#include <cstdint>
#include <climits>

namespace dyphur {

namespace {

// Ray-AABB slab test against fp16 node bounds (promoted to fp32).
// Returns true and updates tmin/tmax if the ray intersects [mn, mx].
// best_t: current closest hit distance for early exit.
inline bool aabb_hit(float ox, float oy, float oz,
                     float inv_dx, float inv_dy, float inv_dz,
                     float mnx, float mny, float mnz,
                     float mxx, float mxy, float mxz,
                     float best_t) {
    float t1x = (mnx - ox) * inv_dx,  t2x = (mxx - ox) * inv_dx;
    float t1y = (mny - oy) * inv_dy,  t2y = (mxy - oy) * inv_dy;
    float t1z = (mnz - oz) * inv_dz,  t2z = (mxz - oz) * inv_dz;

    float tmin = t1x < t2x ? t1x : t2x;
    float tmax = t1x < t2x ? t2x : t1x;

    float ty_lo = t1y < t2y ? t1y : t2y, ty_hi = t1y < t2y ? t2y : t1y;
    tmin = tmin > ty_lo ? tmin : ty_lo;
    tmax = tmax < ty_hi ? tmax : ty_hi;

    float tz_lo = t1z < t2z ? t1z : t2z, tz_hi = t1z < t2z ? t2z : t1z;
    tmin = tmin > tz_lo ? tmin : tz_lo;
    tmax = tmax < tz_hi ? tmax : tz_hi;

    return tmax >= 0.f && tmax >= tmin && tmin < best_t;
}

// Exact OBB intersection for a Box body.  Returns t or -1 on miss.
inline float box_hit(float ox, float oy, float oz,
                     float dx, float dy, float dz,
                     float px, float py, float pz,
                     float rw, float rx, float ry, float rz,
                     float hx, float hy, float hz,
                     float best_t) {
    // Transform ray into body-local frame.
    Quatf q{rw, rx, ry, rz};
    Quatf qi = q.conjugate();
    Vec3f oc{ox - px, oy - py, oz - pz};
    Vec3f lO = qi.rotate(oc);
    Vec3f lD = qi.rotate(Vec3f{dx, dy, dz});

    float inv_lx = 1.f / lD.x;
    float inv_ly = 1.f / lD.y;
    float inv_lz = 1.f / lD.z;

    float t1x = (-hx - lO.x) * inv_lx,  t2x = ( hx - lO.x) * inv_lx;
    float t1y = (-hy - lO.y) * inv_ly,  t2y = ( hy - lO.y) * inv_ly;
    float t1z = (-hz - lO.z) * inv_lz,  t2z = ( hz - lO.z) * inv_lz;

    float tmin = t1x < t2x ? t1x : t2x;
    float tmax = t1x < t2x ? t2x : t1x;

    float ty_lo = t1y < t2y ? t1y : t2y, ty_hi = t1y < t2y ? t2y : t1y;
    tmin = tmin > ty_lo ? tmin : ty_lo;
    tmax = tmax < ty_hi ? tmax : ty_hi;

    float tz_lo = t1z < t2z ? t1z : t2z, tz_hi = t1z < t2z ? t2z : t1z;
    tmin = tmin > tz_lo ? tmin : tz_lo;
    tmax = tmax < tz_hi ? tmax : tz_hi;

    if (tmax < 0.f || tmin > tmax) return -1.f;
    float t = tmin >= 0.f ? tmin : tmax;
    return (t >= 0.f && t < best_t) ? t : -1.f;
}

// Analytic sphere intersection.  Returns t or -1 on miss.
inline float sphere_hit(float ox, float oy, float oz,
                        float dx, float dy, float dz,
                        float cx, float cy, float cz,
                        float r, float best_t) {
    float ocx = ox - cx, ocy = oy - cy, ocz = oz - cz;
    float a   = dx*dx + dy*dy + dz*dz;
    float b   = dx*ocx + dy*ocy + dz*ocz;
    float c   = ocx*ocx + ocy*ocy + ocz*ocz - r*r;
    float disc = b*b - a*c;
    if (disc < 0.f) return -1.f;
    float sq = sycl::sqrt(disc);
    float t  = (-b - sq) / a;
    if (t < 0.f) t = (-b + sq) / a;
    return (t >= 0.f && t < best_t) ? t : -1.f;
}

// Exact intersection test for a single leaf body.  Updates best_t/best_body.
inline void leaf_hit(uint32_t bi,
                     float ox, float oy, float oz,
                     float dx, float dy, float dz,
                     const BodyView& bodies,
                     const ShapeView& shapes,
                     float& best_t, uint32_t& best_body) {
    uint32_t sh    = bodies.shape[bi];
    auto     stype = static_cast<ShapeType>(shapes.type[sh]);
    float    hx    = shapes.half_x[sh];
    float    hy    = shapes.half_y[sh];
    float    hz    = shapes.half_z[sh];

    float t;
    if (stype == ShapeType::Sphere) {
        t = sphere_hit(ox, oy, oz, dx, dy, dz,
                       bodies.pos_x[bi], bodies.pos_y[bi], bodies.pos_z[bi],
                       hx, best_t);
    } else {
        // Box (and ConvexHull/Mesh fall back to OBB for now).
        t = box_hit(ox, oy, oz, dx, dy, dz,
                    bodies.pos_x[bi], bodies.pos_y[bi], bodies.pos_z[bi],
                    bodies.rot_w[bi], bodies.rot_x[bi], bodies.rot_y[bi], bodies.rot_z[bi],
                    hx, hy, hz, best_t);
    }
    if (t >= 0.f) {
        best_t    = t;
        best_body = bi;
    }
}

} // namespace

void ray_query(Stream& s,
               const BvhView&   bvh,
               const BodyView&  bodies,
               const ShapeView& shapes,
               const Ray*        d_rays,
               RayHit*           d_hits,
               uint32_t          n_rays) {
    if (n_rays == 0) return;

    parallel_for(s, static_cast<size_t>(n_rays), [=](size_t idx) {
        const Ray ray = d_rays[idx];

        float best_t    = ray.t_max;
        uint32_t best_body = UINT32_MAX;

        float inv_dx = 1.f / ray.dx;
        float inv_dy = 1.f / ray.dy;
        float inv_dz = 1.f / ray.dz;

        if (bvh.n == 0) {
            // No bodies.
        } else if (bvh.n == 1) {
            // BVH not built for n==1 (build_and_query returns early); sorted_idx
            // is unpopulated. The only body is always index 0.
            leaf_hit(0u, ray.ox, ray.oy, ray.oz, ray.dx, ray.dy, ray.dz,
                     bodies, shapes, best_t, best_body);
        } else {
            // BVH traversal.
            constexpr int kStack = 64;
            int32_t stack[kStack];
            int top = 0;
            stack[top++] = bvh.root[0];

            const uint32_t n_int = bvh.n - 1u;  // first leaf node index == n-1

            while (top > 0) {
                int32_t node = stack[--top];

                // Node AABB test (one packed load, fp16 → fp32).
                NodeBox nb = bvh.aabb[node];
                float mnx = float(nb.mn_x);
                float mny = float(nb.mn_y);
                float mnz = float(nb.mn_z);
                float mxx = float(nb.mx_x);
                float mxy = float(nb.mx_y);
                float mxz = float(nb.mx_z);

                if (!aabb_hit(ray.ox, ray.oy, ray.oz,
                              inv_dx, inv_dy, inv_dz,
                              mnx, mny, mnz, mxx, mxy, mxz, best_t))
                    continue;

                if (static_cast<uint32_t>(node) >= n_int) {
                    // Leaf: exact intersection test.
                    uint32_t bi = bvh.sorted_idx[node - static_cast<int32_t>(n_int)];
                    leaf_hit(bi, ray.ox, ray.oy, ray.oz, ray.dx, ray.dy, ray.dz,
                             bodies, shapes, best_t, best_body);
                } else {
                    // Internal: push children (right first so left is processed first).
                    if (top + 1 < kStack) {
                        stack[top++] = bvh.right[node];
                        stack[top++] = bvh.left[node];
                    }
                }
            }
        }

        d_hits[idx] = RayHit{best_t, best_body};
    });
}

} // namespace dyphur
