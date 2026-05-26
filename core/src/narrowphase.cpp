#include <core/narrowphase.hpp>
#include <compute/atomics.hpp>
#include <compute/kernel.hpp>
#include <sycl/sycl.hpp>

namespace dyphur {

// ── Device helpers ────────────────────────────────────────────────────────────
namespace {

// Emit one contact atomically into the ContactView arrays.
// Returns false if the buffer is full (contact dropped).
inline bool emit(ContactView cv,
                 uint32_t ia, uint32_t ib,
                 float px, float py, float pz,
                 float nx, float ny, float nz,
                 float depth) {
    uint32_t idx = atomic_add_seq(cv.n, 1u);
    if (idx >= cv.capacity) return false;
    cv.body_a[idx] = ia;  cv.body_b[idx] = ib;
    cv.pos_x[idx]  = px;  cv.pos_y[idx]  = py;  cv.pos_z[idx]  = pz;
    cv.norm_x[idx] = nx;  cv.norm_y[idx] = ny;  cv.norm_z[idx] = nz;
    cv.depth[idx]  = depth;
    return true;
}

// ── Sphere–Sphere ─────────────────────────────────────────────────────────────
// Normal points from B toward A.
inline bool sphere_sphere(
    float acx, float acy, float acz, float ra,
    float bcx, float bcy, float bcz, float rb,
    float& px, float& py, float& pz,
    float& nx, float& ny, float& nz,
    float& depth)
{
    float dx = acx - bcx, dy = acy - bcy, dz = acz - bcz;
    float dist2 = dx*dx + dy*dy + dz*dz;
    float sum_r  = ra + rb;
    if (dist2 >= sum_r * sum_r) return false;

    float dist = sycl::sqrt(dist2);
    if (dist < 1e-8f) { nx = 0.f; ny = 1.f; nz = 0.f; }
    else { float inv = 1.f / dist; nx = dx*inv; ny = dy*inv; nz = dz*inv; }

    depth = sum_r - dist;
    // Midpoint between the two surface points.
    float sa = acx - nx*ra, sb = bcx + nx*rb;
    px = 0.5f*(sa + sb);
    sa = acy - ny*ra;  sb = bcy + ny*rb;  py = 0.5f*(sa + sb);
    sa = acz - nz*ra;  sb = bcz + nz*rb;  pz = 0.5f*(sa + sb);
    return true;
}

// ── Sphere–Box ────────────────────────────────────────────────────────────────
// Box A, Sphere B. Normal points from box (A) toward sphere (B)... wait:
// Our convention: normal from B toward A.  A = box, B = sphere → normal from sphere toward box.
// But then the caller that has sphere-first will swap.  Keep consistent:
// HERE: A = box, B = sphere; normal points from B (sphere) toward A (box).
// depth > 0 means sphere penetrates box.
inline bool sphere_box(
    // Sphere
    float scx, float scy, float scz, float sr,
    // Box axes (columns of rotation matrix: world-space local X, Y, Z)
    float bcx, float bcy, float bcz,
    float e0x, float e0y, float e0z,
    float e1x, float e1y, float e1z,
    float e2x, float e2y, float e2z,
    float bh0, float bh1, float bh2,
    float& px, float& py, float& pz,
    float& nx, float& ny, float& nz,
    float& depth)
{
    // d = sphere_center - box_center in world
    float dx = scx - bcx, dy = scy - bcy, dz = scz - bcz;

    // Project onto box-local axes
    float q0 = dx*e0x + dy*e0y + dz*e0z;
    float q1 = dx*e1x + dy*e1y + dz*e1z;
    float q2 = dx*e2x + dy*e2y + dz*e2z;

    // Closest point on box (clamped to half-extents)
    float c0 = q0 < -bh0 ? -bh0 : (q0 > bh0 ? bh0 : q0);
    float c1 = q1 < -bh1 ? -bh1 : (q1 > bh1 ? bh1 : q1);
    float c2 = q2 < -bh2 ? -bh2 : (q2 > bh2 ? bh2 : q2);

    // Vector from closest box point to sphere centre in local space
    float lx = q0 - c0, ly = q1 - c1, lz = q2 - c2;
    float dist2 = lx*lx + ly*ly + lz*lz;

    bool inside = (dist2 < 1e-12f);
    if (inside) {
        // Sphere centre is inside the box: push out through nearest face.
        float pen0 = bh0 - sycl::fabs(q0);
        float pen1 = bh1 - sycl::fabs(q1);
        float pen2 = bh2 - sycl::fabs(q2);
        if (pen0 <= pen1 && pen0 <= pen2) {
            float s = q0 >= 0.f ? 1.f : -1.f;
            nx = -(s*e0x); ny = -(s*e0y); nz = -(s*e0z); // from sphere toward box face
            depth = sr + pen0;
        } else if (pen1 <= pen2) {
            float s = q1 >= 0.f ? 1.f : -1.f;
            nx = -(s*e1x); ny = -(s*e1y); nz = -(s*e1z);
            depth = sr + pen1;
        } else {
            float s = q2 >= 0.f ? 1.f : -1.f;
            nx = -(s*e2x); ny = -(s*e2y); nz = -(s*e2z);
            depth = sr + pen2;
        }
        // Contact point: sphere surface in normal direction
        px = scx - nx*sr; py = scy - ny*sr; pz = scz - nz*sr;
        return true;
    }

    if (dist2 >= sr*sr) return false;

    // Sphere centre outside: closest surface point is the contact.
    float dist   = sycl::sqrt(dist2);
    float inv_d  = 1.f / dist;
    // lx,ly,lz is local-frame; convert to world
    float wx = lx*e0x + ly*e1x + lz*e2x;
    float wy = lx*e0y + ly*e1y + lz*e2y;
    float wz = lx*e0z + ly*e1z + lz*e2z;
    // Normal from sphere (B) toward box (A) = -(sphere→box) = -(wx,wy,wz)/dist
    nx = -wx*inv_d; ny = -wy*inv_d; nz = -wz*inv_d;
    depth = sr - dist;
    // Closest point on box surface in world
    px = bcx + c0*e0x + c1*e1x + c2*e2x;
    py = bcy + c0*e0y + c1*e1y + c2*e2y;
    pz = bcz + c0*e0z + c1*e1z + c2*e2z;
    return true;
}

// ── Box–Box (SAT, vertex-face manifold) ───────────────────────────────────────
// A and B are boxes. Normal points from B toward A.
// Returns number of contacts emitted (0–4).
//
// Axes layout: e[axis][xyz], h[axis].
static int box_box(
    float acx, float acy, float acz,
    const float Ae[3][3], const float Ah[3],
    float bcx, float bcy, float bcz,
    const float Be[3][3], const float Bh[3],
    float out_px[4], float out_py[4], float out_pz[4],
    float& out_nx, float& out_ny, float& out_nz,
    float out_d[4])
{
    const float eps_par = 1e-5f;  // parallel-axis threshold for AbsC

    // Precompute axis dot-products and d components
    float C[3][3], AbsC[3][3];
    float dAB[3], dBA[3]; // (B-A) projected onto A's and B's axes
    float dx = bcx-acx, dy = bcy-acy, dz = bcz-acz;
    for (int i = 0; i < 3; ++i) {
        dAB[i] = dx*Ae[i][0] + dy*Ae[i][1] + dz*Ae[i][2];
        dBA[i] = dx*Be[i][0] + dy*Be[i][1] + dz*Be[i][2];
        for (int j = 0; j < 3; ++j) {
            C[i][j]    = Ae[i][0]*Be[j][0] + Ae[i][1]*Be[j][1] + Ae[i][2]*Be[j][2];
            AbsC[i][j] = sycl::fabs(C[i][j]) + eps_par;
        }
    }

    float best_ov  = 1e30f;
    int   best_type = -1; // 0=A face, 1=B face, 2=edge
    int   best_i   = 0, best_j = 0;
    float best_nx  = 0.f, best_ny = 1.f, best_nz = 0.f;

    // Helper: test one axis, update best.
    // axis must have unit length; lx,ly,lz are its components.
    // Returns false if separated (caller returns 0).
#define SAT_TEST(lx_, ly_, lz_, type_, i_, j_)                                   \
    do {                                                                           \
        float rA = Ah[0]*sycl::fabs((lx_)*Ae[0][0]+(ly_)*Ae[0][1]+(lz_)*Ae[0][2]) \
                 + Ah[1]*sycl::fabs((lx_)*Ae[1][0]+(ly_)*Ae[1][1]+(lz_)*Ae[1][2]) \
                 + Ah[2]*sycl::fabs((lx_)*Ae[2][0]+(ly_)*Ae[2][1]+(lz_)*Ae[2][2]); \
        float rB = Bh[0]*sycl::fabs((lx_)*Be[0][0]+(ly_)*Be[0][1]+(lz_)*Be[0][2]) \
                 + Bh[1]*sycl::fabs((lx_)*Be[1][0]+(ly_)*Be[1][1]+(lz_)*Be[1][2]) \
                 + Bh[2]*sycl::fabs((lx_)*Be[2][0]+(ly_)*Be[2][1]+(lz_)*Be[2][2]); \
        float t  = sycl::fabs(dx*(lx_) + dy*(ly_) + dz*(lz_));                    \
        float ov = rA + rB - t;                                                    \
        if (ov <= 0.f) return 0;                                                   \
        if (ov < best_ov) {                                                        \
            best_ov   = ov;                                                        \
            best_type = (type_); best_i = (i_); best_j = (j_);                    \
            float sgn = (dx*(lx_)+dy*(ly_)+dz*(lz_)) >= 0.f ? -1.f : 1.f;        \
            best_nx = sgn*(lx_); best_ny = sgn*(ly_); best_nz = sgn*(lz_);        \
        }                                                                          \
    } while(false)

    // Face axes of A
    SAT_TEST(Ae[0][0], Ae[0][1], Ae[0][2], 0, 0, 0);
    SAT_TEST(Ae[1][0], Ae[1][1], Ae[1][2], 0, 1, 0);
    SAT_TEST(Ae[2][0], Ae[2][1], Ae[2][2], 0, 2, 0);
    // Face axes of B
    SAT_TEST(Be[0][0], Be[0][1], Be[0][2], 1, 0, 0);
    SAT_TEST(Be[1][0], Be[1][1], Be[1][2], 1, 1, 0);
    SAT_TEST(Be[2][0], Be[2][1], Be[2][2], 1, 2, 0);
    // Edge-edge cross products
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float cx_ = Ae[i][1]*Be[j][2] - Ae[i][2]*Be[j][1];
            float cy_ = Ae[i][2]*Be[j][0] - Ae[i][0]*Be[j][2];
            float cz_ = Ae[i][0]*Be[j][1] - Ae[i][1]*Be[j][0];
            float len2 = cx_*cx_ + cy_*cy_ + cz_*cz_;
            if (len2 < eps_par*eps_par) continue; // parallel, skip
            float inv_l = sycl::rsqrt(len2);
            float lx_ = cx_*inv_l, ly_ = cy_*inv_l, lz_ = cz_*inv_l;
            // Custom SAT_TEST that doesn't return on separation from within loop
            {
                float rA = Ah[0]*sycl::fabs(lx_*Ae[0][0]+ly_*Ae[0][1]+lz_*Ae[0][2])
                         + Ah[1]*sycl::fabs(lx_*Ae[1][0]+ly_*Ae[1][1]+lz_*Ae[1][2])
                         + Ah[2]*sycl::fabs(lx_*Ae[2][0]+ly_*Ae[2][1]+lz_*Ae[2][2]);
                float rB = Bh[0]*sycl::fabs(lx_*Be[0][0]+ly_*Be[0][1]+lz_*Be[0][2])
                         + Bh[1]*sycl::fabs(lx_*Be[1][0]+ly_*Be[1][1]+lz_*Be[1][2])
                         + Bh[2]*sycl::fabs(lx_*Be[2][0]+ly_*Be[2][1]+lz_*Be[2][2]);
                float t  = sycl::fabs(dx*lx_ + dy*ly_ + dz*lz_);
                float ov = rA + rB - t;
                if (ov <= 0.f) return 0;
                if (ov < best_ov) {
                    best_ov   = ov;
                    best_type = 2; best_i = i; best_j = j;
                    float sgn = (dx*lx_+dy*ly_+dz*lz_) >= 0.f ? -1.f : 1.f;
                    best_nx = sgn*lx_; best_ny = sgn*ly_; best_nz = sgn*lz_;
                }
            }
        }
    }
#undef SAT_TEST

    out_nx = best_nx; out_ny = best_ny; out_nz = best_nz;

    int n_out = 0;

    if (best_type == 0 || best_type == 1) {
        // Face contact: prefer the larger face as reference so that the smaller
        // (incident) box's vertices lie within the reference face bounds.
        // When best_type==0 (A face selected), check if B's parallel face is
        // larger; if so, swap to use B as reference and iterate A's vertices.
        if (best_type == 0) {
            int t0A = (best_i+1)%3, t1A = (best_i+2)%3;
            // Find B's axis most aligned with A's ref axis
            int j_best = 0;
            for (int j = 1; j < 3; ++j)
                if (sycl::fabs(C[best_i][j]) > sycl::fabs(C[best_i][j_best])) j_best = j;
            int t0B = (j_best+1)%3, t1B = (j_best+2)%3;
            if (Bh[t0B]*Bh[t1B] > Ah[t0A]*Ah[t1A])
                best_type = 1, best_i = j_best; // B has larger face → B is reference
        }

        bool a_is_ref = (best_type == 0);
        int  ref_ax   = best_i;

        const float* ref_cx_p = a_is_ref ? &acx : &bcx;
        const float* ref_cy_p = a_is_ref ? &acy : &bcy;
        const float* ref_cz_p = a_is_ref ? &acz : &bcz;
        float rcx = *ref_cx_p, rcy = *ref_cy_p, rcz = *ref_cz_p;
        (void)rcz; // suppress unused warning

        const float (*Re)[3] = a_is_ref ? Ae : Be;
        const float  *Rh     = a_is_ref ? Ah : Bh;
        const float (*Ie)[3] = a_is_ref ? Be : Ae;
        const float  *Ih     = a_is_ref ? Bh : Ah;
        float icx = a_is_ref ? bcx : acx;
        float icy = a_is_ref ? bcy : acy;
        float icz = a_is_ref ? bcz : acz;

        // Tangent axes of the reference face
        int t0 = (ref_ax + 1) % 3, t1 = (ref_ax + 2) % 3;

        // Sign: N points from B toward A.
        // For ref=A face: N = best_n (from B→A).  The reference face
        // faces B, so the face outward (toward B) = -N.
        // depth formula for incident vertex v:
        //   depth_v = Rh[ref_ax] + (v - ref_center) · N  (a_is_ref)
        //   depth_v = Rh[ref_ax] - (v - ref_center) · N  (!a_is_ref)
        float rc_dot_n = rcx*best_nx + rcy*best_ny + rcz*best_nz;  // R.center · N
        float sign_depth = a_is_ref ? 1.f : -1.f;

        for (int k = 0; k < 8; ++k) {
            float sx = (k & 1) ? 1.f : -1.f;
            float sy = (k & 2) ? 1.f : -1.f;
            float sz = (k & 4) ? 1.f : -1.f;
            float vx = icx + sx*Ih[0]*Ie[0][0] + sy*Ih[1]*Ie[1][0] + sz*Ih[2]*Ie[2][0];
            float vy = icy + sx*Ih[0]*Ie[0][1] + sy*Ih[1]*Ie[1][1] + sz*Ih[2]*Ie[2][1];
            float vz = icz + sx*Ih[0]*Ie[0][2] + sy*Ih[1]*Ie[1][2] + sz*Ih[2]*Ie[2][2];

            // Penetration depth
            float v_dot_n = vx*best_nx + vy*best_ny + vz*best_nz;
            float dv = Rh[ref_ax] + sign_depth * (v_dot_n - rc_dot_n);
            if (dv <= 0.f) continue;

            // Within-face bounds check on ref's tangent axes
            float vx_rel = vx - (a_is_ref ? acx : bcx);
            float vy_rel = vy - (a_is_ref ? acy : bcy);
            float vz_rel = vz - (a_is_ref ? acz : bcz);
            float qt0 = vx_rel*Re[t0][0] + vy_rel*Re[t0][1] + vz_rel*Re[t0][2];
            float qt1 = vx_rel*Re[t1][0] + vy_rel*Re[t1][1] + vz_rel*Re[t1][2];
            if (sycl::fabs(qt0) > Rh[t0] + 1e-3f) continue;
            if (sycl::fabs(qt1) > Rh[t1] + 1e-3f) continue;

            // Project vertex onto reference face plane
            float cp_x = vx - sign_depth * dv * best_nx;
            float cp_y = vy - sign_depth * dv * best_ny;
            float cp_z = vz - sign_depth * dv * best_nz;

            if (n_out < 4) {
                out_px[n_out] = cp_x; out_py[n_out] = cp_y; out_pz[n_out] = cp_z;
                out_d[n_out]  = dv;
                ++n_out;
            }
        }
    } else {
        // Edge–edge: one contact at the closest point of the two edge lines.
        int i = best_i, j = best_j;

        // Pick the supporting edge of A along Ae[i] (the one facing toward B)
        int t0A = (i+1)%3, t1A = (i+2)%3;
        float s0A = (-best_nx*Ae[t0A][0] - best_ny*Ae[t0A][1] - best_nz*Ae[t0A][2]) >= 0.f ? 1.f : -1.f;
        float s1A = (-best_nx*Ae[t1A][0] - best_ny*Ae[t1A][1] - best_nz*Ae[t1A][2]) >= 0.f ? 1.f : -1.f;
        float pAx = acx + s0A*Ah[t0A]*Ae[t0A][0] + s1A*Ah[t1A]*Ae[t1A][0];
        float pAy = acy + s0A*Ah[t0A]*Ae[t0A][1] + s1A*Ah[t1A]*Ae[t1A][1];
        float pAz = acz + s0A*Ah[t0A]*Ae[t0A][2] + s1A*Ah[t1A]*Ae[t1A][2];

        // Pick the supporting edge of B along Be[j] (facing toward A)
        int t0B = (j+1)%3, t1B = (j+2)%3;
        float s0B = (best_nx*Be[t0B][0] + best_ny*Be[t0B][1] + best_nz*Be[t0B][2]) >= 0.f ? 1.f : -1.f;
        float s1B = (best_nx*Be[t1B][0] + best_ny*Be[t1B][1] + best_nz*Be[t1B][2]) >= 0.f ? 1.f : -1.f;
        float pBx = bcx + s0B*Bh[t0B]*Be[t0B][0] + s1B*Bh[t1B]*Be[t1B][0];
        float pBy = bcy + s0B*Bh[t0B]*Be[t0B][1] + s1B*Bh[t1B]*Be[t1B][1];
        float pBz = bcz + s0B*Bh[t0B]*Be[t0B][2] + s1B*Bh[t1B]*Be[t1B][2];

        // Closest point on two infinite lines:
        // Line A: pA + s * Ae[i];   Line B: pB + t * Be[j]
        float uAx = Ae[i][0], uAy = Ae[i][1], uAz = Ae[i][2];
        float uBx = Be[j][0], uBy = Be[j][1], uBz = Be[j][2];
        float b12 = C[i][j]; // uA · uB (already computed)
        float denom = 1.f - b12*b12;

        float wx = pAx - pBx, wy = pAy - pBy, wz = pAz - pBz;
        float d1 = wx*uAx + wy*uAy + wz*uAz;
        float d2 = wx*uBx + wy*uBy + wz*uBz;

        float s_t = 0.f, t_t = 0.f;
        if (sycl::fabs(denom) > 1e-6f) {
            s_t = (b12*d2 - d1) / denom;
            t_t = (d2 - b12*d1) / denom;
        }

        float qAx = pAx + s_t*uAx, qAy = pAy + s_t*uAy, qAz = pAz + s_t*uAz;
        float qBx = pBx + t_t*uBx, qBy = pBy + t_t*uBy, qBz = pBz + t_t*uBz;

        out_px[0] = 0.5f*(qAx + qBx);
        out_py[0] = 0.5f*(qAy + qBy);
        out_pz[0] = 0.5f*(qAz + qBz);
        out_d[0]  = best_ov;
        n_out = 1;
    }

    return n_out;
}

} // anonymous namespace

// ── Narrowphase constructor ───────────────────────────────────────────────────

Narrowphase::Narrowphase(Stream& s, uint32_t max_contacts)
    : store_(s, max_contacts)
{}

// ── Narrowphase::run ──────────────────────────────────────────────────────────

void Narrowphase::run(Stream& s,
                      const ContactPair* d_pairs, uint32_t n_pairs,
                      const BodyView& bodies, const ShapeView& shapes)
{
    store_.reset(s);
    if (n_pairs == 0) return;

    ContactView cv = store_.view();
    const BodyView  bv = bodies;
    const ShapeView sv = shapes;
    const uint32_t  np = n_pairs;

    // Sequential single work-item: pairs processed in fixed order → deterministic
    // contact sequence when pairs are pre-sorted by (a,b).
    parallel_for(s, 1, [=](size_t) {
      for (uint32_t idx = 0; idx < np; ++idx) {
        ContactPair pair = d_pairs[idx];
        uint32_t ia = pair.a, ib = pair.b;

        uint32_t sha = bv.shape[ia], shb = bv.shape[ib];
        uint32_t ta  = sv.type[sha], tb  = sv.type[shb];

        constexpr uint32_t kBox    = static_cast<uint32_t>(ShapeType::Box);
        constexpr uint32_t kSphere = static_cast<uint32_t>(ShapeType::Sphere);

        // Rotation matrix for a body: columns are local axes in world space.
        // Ae[axis][xyz component], e.g. Ae[0] = local-X in world.
        auto make_axes = [&](uint32_t bi, float Ae[3][3]) {
            float qw = bv.rot_w[bi], qx = bv.rot_x[bi];
            float qy = bv.rot_y[bi], qz = bv.rot_z[bi];
            Ae[0][0] = 1.f-2.f*(qy*qy+qz*qz); Ae[0][1] = 2.f*(qx*qy+qz*qw); Ae[0][2] = 2.f*(qx*qz-qy*qw);
            Ae[1][0] = 2.f*(qx*qy-qz*qw);      Ae[1][1] = 1.f-2.f*(qx*qx+qz*qz); Ae[1][2] = 2.f*(qy*qz+qx*qw);
            Ae[2][0] = 2.f*(qx*qz+qy*qw);      Ae[2][1] = 2.f*(qy*qz-qx*qw); Ae[2][2] = 1.f-2.f*(qx*qx+qy*qy);
        };

        float px, py, pz, nx, ny, nz, depth;

        if (ta == kSphere && tb == kSphere) {
            if (!sphere_sphere(bv.pos_x[ia], bv.pos_y[ia], bv.pos_z[ia], sv.half_x[sha],
                               bv.pos_x[ib], bv.pos_y[ib], bv.pos_z[ib], sv.half_x[shb],
                               px, py, pz, nx, ny, nz, depth)) continue;
            emit(cv, ia, ib, px, py, pz, nx, ny, nz, depth);

        } else if (ta == kSphere && tb == kBox) {
            float Be[3][3]; float Bh[3] = {sv.half_x[shb], sv.half_y[shb], sv.half_z[shb]};
            make_axes(ib, Be);
            if (!sphere_box(bv.pos_x[ia], bv.pos_y[ia], bv.pos_z[ia], sv.half_x[sha],
                            bv.pos_x[ib], bv.pos_y[ib], bv.pos_z[ib],
                            Be[0][0], Be[0][1], Be[0][2],
                            Be[1][0], Be[1][1], Be[1][2],
                            Be[2][0], Be[2][1], Be[2][2],
                            Bh[0], Bh[1], Bh[2],
                            px, py, pz, nx, ny, nz, depth)) continue;
            nx = -nx; ny = -ny; nz = -nz;
            emit(cv, ia, ib, px, py, pz, nx, ny, nz, depth);

        } else if (ta == kBox && tb == kSphere) {
            float Ae[3][3]; float Ah[3] = {sv.half_x[sha], sv.half_y[sha], sv.half_z[sha]};
            make_axes(ia, Ae);
            if (!sphere_box(bv.pos_x[ib], bv.pos_y[ib], bv.pos_z[ib], sv.half_x[shb],
                            bv.pos_x[ia], bv.pos_y[ia], bv.pos_z[ia],
                            Ae[0][0], Ae[0][1], Ae[0][2],
                            Ae[1][0], Ae[1][1], Ae[1][2],
                            Ae[2][0], Ae[2][1], Ae[2][2],
                            Ah[0], Ah[1], Ah[2],
                            px, py, pz, nx, ny, nz, depth)) continue;
            emit(cv, ia, ib, px, py, pz, nx, ny, nz, depth);

        } else if (ta == kBox && tb == kBox) {
            float Ae[3][3]; float Ah[3] = {sv.half_x[sha], sv.half_y[sha], sv.half_z[sha]};
            float Be_m[3][3]; float Bh[3] = {sv.half_x[shb], sv.half_y[shb], sv.half_z[shb]};
            make_axes(ia, Ae);
            make_axes(ib, Be_m);

            float opx[4], opy[4], opz[4], od[4];
            float onx, ony, onz;
            int nc = box_box(bv.pos_x[ia], bv.pos_y[ia], bv.pos_z[ia], Ae, Ah,
                             bv.pos_x[ib], bv.pos_y[ib], bv.pos_z[ib], Be_m, Bh,
                             opx, opy, opz, onx, ony, onz, od);
            for (int k = 0; k < nc; ++k)
                emit(cv, ia, ib, opx[k], opy[k], opz[k], onx, ony, onz, od[k]);
        }
      } // for idx
    }); // parallel_for
}


} // namespace dyphur
