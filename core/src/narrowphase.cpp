#include <core/narrowphase.hpp>
#include <compute/atomics.hpp>
#include <compute/kernel.hpp>
#include <compute/sort.hpp>
#include <sycl/sycl.hpp>

namespace dyphur {

// ── Device helpers ────────────────────────────────────────────────────────────
namespace {

inline uint32_t next_pow2(uint32_t n) {
    if (n <= 1) return 1;
    --n; n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16;
    return n + 1;
}

// Per-pair contact sink: contacts are appended to a scratch ContactView and each
// is tagged with a sort key = (pair_idx << 32) | sub_idx, where sub_idx counts the
// emissions within this pair. Sorting by that key restores the exact order a
// sequential (pair 0, pair 1, …) pass would emit.
struct PairSink {
    ContactView cv;
    uint64_t*   keys;
    uint32_t    pair_idx;
    uint32_t    sub;
};

inline bool emit(PairSink& sk,
                 uint32_t ia, uint32_t ib,
                 float px, float py, float pz,
                 float nx, float ny, float nz,
                 float depth) {
    uint32_t idx = atomic_add_seq(sk.cv.n, 1u);
    if (idx >= sk.cv.capacity) return false;
    sk.cv.body_a[idx] = ia;  sk.cv.body_b[idx] = ib;
    sk.cv.pos_x[idx]  = px;  sk.cv.pos_y[idx]  = py;  sk.cv.pos_z[idx]  = pz;
    sk.cv.norm_x[idx] = nx;  sk.cv.norm_y[idx] = ny;  sk.cv.norm_z[idx] = nz;
    sk.cv.depth[idx]  = depth;
    sk.keys[idx] = (static_cast<uint64_t>(sk.pair_idx) << 32) | sk.sub;
    ++sk.sub;
    return true;
}

// ── Sphere–Sphere ─────────────────────────────────────────────────────────────
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
    float sa = acx - nx*ra, sb = bcx + nx*rb;
    px = 0.5f*(sa + sb);
    sa = acy - ny*ra;  sb = bcy + ny*rb;  py = 0.5f*(sa + sb);
    sa = acz - nz*ra;  sb = bcz + nz*rb;  pz = 0.5f*(sa + sb);
    return true;
}

// ── Sphere–Box ────────────────────────────────────────────────────────────────
inline bool sphere_box(
    float scx, float scy, float scz, float sr,
    float bcx, float bcy, float bcz,
    float e0x, float e0y, float e0z,
    float e1x, float e1y, float e1z,
    float e2x, float e2y, float e2z,
    float bh0, float bh1, float bh2,
    float& px, float& py, float& pz,
    float& nx, float& ny, float& nz,
    float& depth)
{
    float dx = scx - bcx, dy = scy - bcy, dz = scz - bcz;
    float q0 = dx*e0x + dy*e0y + dz*e0z;
    float q1 = dx*e1x + dy*e1y + dz*e1z;
    float q2 = dx*e2x + dy*e2y + dz*e2z;
    float c0 = q0 < -bh0 ? -bh0 : (q0 > bh0 ? bh0 : q0);
    float c1 = q1 < -bh1 ? -bh1 : (q1 > bh1 ? bh1 : q1);
    float c2 = q2 < -bh2 ? -bh2 : (q2 > bh2 ? bh2 : q2);
    float lx = q0 - c0, ly = q1 - c1, lz = q2 - c2;
    float dist2 = lx*lx + ly*ly + lz*lz;

    bool inside = (dist2 < 1e-12f);
    if (inside) {
        float pen0 = bh0 - sycl::fabs(q0);
        float pen1 = bh1 - sycl::fabs(q1);
        float pen2 = bh2 - sycl::fabs(q2);
        if (pen0 <= pen1 && pen0 <= pen2) {
            float s = q0 >= 0.f ? 1.f : -1.f;
            nx = -(s*e0x); ny = -(s*e0y); nz = -(s*e0z);
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
        px = scx - nx*sr; py = scy - ny*sr; pz = scz - nz*sr;
        return true;
    }

    if (dist2 >= sr*sr) return false;

    float dist   = sycl::sqrt(dist2);
    float inv_d  = 1.f / dist;
    float wx = lx*e0x + ly*e1x + lz*e2x;
    float wy = lx*e0y + ly*e1y + lz*e2y;
    float wz = lx*e0z + ly*e1z + lz*e2z;
    nx = -wx*inv_d; ny = -wy*inv_d; nz = -wz*inv_d;
    depth = sr - dist;
    px = bcx + c0*e0x + c1*e1x + c2*e2x;
    py = bcy + c0*e0y + c1*e1y + c2*e2y;
    pz = bcz + c0*e0z + c1*e1z + c2*e2z;
    return true;
}

// ── Box–Box (SAT, vertex-face manifold) ───────────────────────────────────────
static int box_box(
    float acx, float acy, float acz,
    const float Ae[3][3], const float Ah[3],
    float bcx, float bcy, float bcz,
    const float Be[3][3], const float Bh[3],
    float out_px[4], float out_py[4], float out_pz[4],
    float& out_nx, float& out_ny, float& out_nz,
    float out_d[4])
{
    const float eps_par = 1e-5f;
    float C[3][3], AbsC[3][3];
    float dAB[3], dBA[3];
    float dx = bcx-acx, dy = bcy-acy, dz = bcz-acz;
    for (int i = 0; i < 3; ++i) {
        dAB[i] = dx*Ae[i][0] + dy*Ae[i][1] + dz*Ae[i][2];
        dBA[i] = dx*Be[i][0] + dy*Be[i][1] + dz*Be[i][2];
        for (int j = 0; j < 3; ++j) {
            C[i][j]    = Ae[i][0]*Be[j][0] + Ae[i][1]*Be[j][1] + Ae[i][2]*Be[j][2];
            AbsC[i][j] = sycl::fabs(C[i][j]) + eps_par;
        }
    }
    (void)dBA; (void)AbsC;

    float best_ov  = 1e30f;
    int   best_type = -1;
    int   best_i   = 0, best_j = 0;
    float best_nx  = 0.f, best_ny = 1.f, best_nz = 0.f;

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

    SAT_TEST(Ae[0][0], Ae[0][1], Ae[0][2], 0, 0, 0);
    SAT_TEST(Ae[1][0], Ae[1][1], Ae[1][2], 0, 1, 0);
    SAT_TEST(Ae[2][0], Ae[2][1], Ae[2][2], 0, 2, 0);
    SAT_TEST(Be[0][0], Be[0][1], Be[0][2], 1, 0, 0);
    SAT_TEST(Be[1][0], Be[1][1], Be[1][2], 1, 1, 0);
    SAT_TEST(Be[2][0], Be[2][1], Be[2][2], 1, 2, 0);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            float cx_ = Ae[i][1]*Be[j][2] - Ae[i][2]*Be[j][1];
            float cy_ = Ae[i][2]*Be[j][0] - Ae[i][0]*Be[j][2];
            float cz_ = Ae[i][0]*Be[j][1] - Ae[i][1]*Be[j][0];
            float len2 = cx_*cx_ + cy_*cy_ + cz_*cz_;
            if (len2 < eps_par*eps_par) continue;
            float inv_l = sycl::rsqrt(len2);
            float lx_ = cx_*inv_l, ly_ = cy_*inv_l, lz_ = cz_*inv_l;
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
        if (best_type == 0) {
            int t0A = (best_i+1)%3, t1A = (best_i+2)%3;
            int j_best = 0;
            for (int j = 1; j < 3; ++j)
                if (sycl::fabs(C[best_i][j]) > sycl::fabs(C[best_i][j_best])) j_best = j;
            int t0B = (j_best+1)%3, t1B = (j_best+2)%3;
            if (Bh[t0B]*Bh[t1B] > Ah[t0A]*Ah[t1A])
                best_type = 1, best_i = j_best;
        }

        bool a_is_ref = (best_type == 0);
        int  ref_ax   = best_i;
        float rcx = a_is_ref ? acx : bcx;
        float rcy = a_is_ref ? acy : bcy;
        float rcz_unused = a_is_ref ? acz : bcz; (void)rcz_unused;
        const float (*Re)[3] = a_is_ref ? Ae : Be;
        const float  *Rh     = a_is_ref ? Ah : Bh;
        const float (*Ie)[3] = a_is_ref ? Be : Ae;
        const float  *Ih     = a_is_ref ? Bh : Ah;
        float icx = a_is_ref ? bcx : acx;
        float icy = a_is_ref ? bcy : acy;
        float icz = a_is_ref ? bcz : acz;
        int t0 = (ref_ax + 1) % 3, t1 = (ref_ax + 2) % 3;
        float rc_dot_n = rcx*best_nx + rcy*best_ny;  // z not needed — implicit in depth check
        (void)rc_dot_n;
        float rc_dot_n3 = rcx*best_nx + rcy*best_ny + (a_is_ref ? acz : bcz)*best_nz;
        float sign_depth = a_is_ref ? 1.f : -1.f;

        for (int k = 0; k < 8; ++k) {
            float sx = (k & 1) ? 1.f : -1.f;
            float sy = (k & 2) ? 1.f : -1.f;
            float sz = (k & 4) ? 1.f : -1.f;
            float vx = icx + sx*Ih[0]*Ie[0][0] + sy*Ih[1]*Ie[1][0] + sz*Ih[2]*Ie[2][0];
            float vy = icy + sx*Ih[0]*Ie[0][1] + sy*Ih[1]*Ie[1][1] + sz*Ih[2]*Ie[2][1];
            float vz = icz + sx*Ih[0]*Ie[0][2] + sy*Ih[1]*Ie[1][2] + sz*Ih[2]*Ie[2][2];
            float v_dot_n = vx*best_nx + vy*best_ny + vz*best_nz;
            float dv = Rh[ref_ax] + sign_depth * (v_dot_n - rc_dot_n3);
            if (dv <= 0.f) continue;
            float vx_rel = vx - (a_is_ref ? acx : bcx);
            float vy_rel = vy - (a_is_ref ? acy : bcy);
            float vz_rel = vz - (a_is_ref ? acz : bcz);
            float qt0 = vx_rel*Re[t0][0] + vy_rel*Re[t0][1] + vz_rel*Re[t0][2];
            float qt1 = vx_rel*Re[t1][0] + vy_rel*Re[t1][1] + vz_rel*Re[t1][2];
            if (sycl::fabs(qt0) > Rh[t0] + 1e-3f) continue;
            if (sycl::fabs(qt1) > Rh[t1] + 1e-3f) continue;
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
        int i = best_i, j = best_j;
        int t0A = (i+1)%3, t1A = (i+2)%3;
        float s0A = (-best_nx*Ae[t0A][0] - best_ny*Ae[t0A][1] - best_nz*Ae[t0A][2]) >= 0.f ? 1.f : -1.f;
        float s1A = (-best_nx*Ae[t1A][0] - best_ny*Ae[t1A][1] - best_nz*Ae[t1A][2]) >= 0.f ? 1.f : -1.f;
        float pAx = acx + s0A*Ah[t0A]*Ae[t0A][0] + s1A*Ah[t1A]*Ae[t1A][0];
        float pAy = acy + s0A*Ah[t0A]*Ae[t0A][1] + s1A*Ah[t1A]*Ae[t1A][1];
        float pAz = acz + s0A*Ah[t0A]*Ae[t0A][2] + s1A*Ah[t1A]*Ae[t1A][2];
        int t0B = (j+1)%3, t1B = (j+2)%3;
        float s0B = (best_nx*Be[t0B][0] + best_ny*Be[t0B][1] + best_nz*Be[t0B][2]) >= 0.f ? 1.f : -1.f;
        float s1B = (best_nx*Be[t1B][0] + best_ny*Be[t1B][1] + best_nz*Be[t1B][2]) >= 0.f ? 1.f : -1.f;
        float pBx = bcx + s0B*Bh[t0B]*Be[t0B][0] + s1B*Bh[t1B]*Be[t1B][0];
        float pBy = bcy + s0B*Bh[t0B]*Be[t0B][1] + s1B*Bh[t1B]*Be[t1B][1];
        float pBz = bcz + s0B*Bh[t0B]*Be[t0B][2] + s1B*Bh[t1B]*Be[t1B][2];
        float uAx = Ae[i][0], uAy = Ae[i][1], uAz = Ae[i][2];
        float uBx = Be[j][0], uBy = Be[j][1], uBz = Be[j][2];
        float b12 = C[i][j];
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

// ── GJK + EPA for convex hull pairs ───────────────────────────────────────────
//
// Convention: simplex v[0] = newest. AO = origin - A = -v[0].
// The support function transforms direction to local space, finds farthest vertex,
// then transforms back to world space using the body's pose.

struct GjkPt {
    float x, y, z;     // on Minkowski difference (world space)
    float ax, ay, az;  // on hull A (world space)
    float bx, by, bz;  // on hull B (world space)
};

struct GjkSimplex {
    GjkPt v[4];
    int   n = 0;
    inline void push(GjkPt p) {
        v[3]=v[2]; v[2]=v[1]; v[1]=v[0]; v[0]=p;
        n = (n < 4) ? n+1 : 4;
    }
};

// World-space support for a convex hull body along direction (dx,dy,dz).
inline GjkPt hull_support_world(
    const ConvexHullView& cv, uint32_t hull_id,
    float px, float py, float pz,   // body position
    float qw, float qx, float qy, float qz, // body rotation
    float dx, float dy, float dz)    // direction in world space
{
    // Transform direction to body-local space: d_local = R^T * d_world
    float rl0 = dx*(1-2*(qy*qy+qz*qz)) + dy*2*(qx*qy+qz*qw) + dz*2*(qx*qz-qy*qw);
    float rl1 = dx*2*(qx*qy-qz*qw) + dy*(1-2*(qx*qx+qz*qz)) + dz*2*(qy*qz+qx*qw);
    float rl2 = dx*2*(qx*qz+qy*qw) + dy*2*(qy*qz-qx*qw) + dz*(1-2*(qx*qx+qy*qy));

    uint32_t s = cv.hull_start[hull_id], n = cv.hull_count[hull_id];
    float best = -1e30f; uint32_t bi = s;
    for (uint32_t i = s; i < s+n; ++i) {
        float d = cv.vtx_x[i]*rl0 + cv.vtx_y[i]*rl1 + cv.vtx_z[i]*rl2;
        if (d > best) { best = d; bi = i; }
    }

    // Transform to world space: v_world = R * v_local + pos
    float vx = cv.vtx_x[bi], vy = cv.vtx_y[bi], vz = cv.vtx_z[bi];
    GjkPt p;
    p.ax = px + vx*(1-2*(qy*qy+qz*qz)) + vy*2*(qx*qy-qz*qw) + vz*2*(qx*qz+qy*qw);
    p.ay = py + vx*2*(qx*qy+qz*qw) + vy*(1-2*(qx*qx+qz*qz)) + vz*2*(qy*qz-qx*qw);
    p.az = pz + vx*2*(qx*qz-qy*qw) + vy*2*(qy*qz+qx*qw) + vz*(1-2*(qx*qx+qy*qy));
    return p;
}

// Minkowski difference support: sup_A(d) - sup_B(-d)
inline GjkPt minkdiff(
    const ConvexHullView& ca, uint32_t ia,
    float apx, float apy, float apz, float aqw, float aqx, float aqy, float aqz,
    const ConvexHullView& cb, uint32_t ib,
    float bpx, float bpy, float bpz, float bqw, float bqx, float bqy, float bqz,
    float dx, float dy, float dz)
{
    GjkPt sa = hull_support_world(ca, ia, apx, apy, apz, aqw, aqx, aqy, aqz, dx, dy, dz);
    GjkPt sb = hull_support_world(cb, ib, bpx, bpy, bpz, bqw, bqx, bqy, bqz, -dx, -dy, -dz);
    GjkPt r;
    r.ax=sa.ax; r.ay=sa.ay; r.az=sa.az;
    r.bx=sb.ax; r.by=sb.ay; r.bz=sb.az; // sb.a* are the support points on B
    r.x=r.ax-r.bx; r.y=r.ay-r.by; r.z=r.az-r.bz;
    return r;
}

// Signed triple product: (AB × AC) · AD
inline float triple(float abx,float aby,float abz, float acx,float acy,float acz, float adx,float ady,float adz)
{
    return (aby*acz-abz*acy)*adx + (abz*acx-abx*acz)*ady + (abx*acy-aby*acx)*adz;
}

// GJK nearest simplex for a line (n==2). Mutates simplex and dir.
// A=v[0] (newest), B=v[1].
inline bool gjk_line(GjkSimplex& s, float& dx, float& dy, float& dz)
{
    float ax=s.v[0].x, ay=s.v[0].y, az=s.v[0].z;
    float bx=s.v[1].x, by=s.v[1].y, bz=s.v[1].z;
    float abx=bx-ax, aby=by-ay, abz=bz-az;
    float aox=-ax,    aoy=-ay,   aoz=-az;
    if (abx*aox+aby*aoy+abz*aoz > 0.f) {
        // Origin projects onto edge AB: dir = triple-product AB×AO×AB
        float t = abx*aox+aby*aoy+abz*aoz;
        float q = abx*abx+aby*aby+abz*abz;
        dx = aox*q - abx*t;
        dy = aoy*q - aby*t;
        dz = aoz*q - abz*t;
    } else {
        s.v[0]=s.v[1]; s.n=1; // keep only A
        dx=aox; dy=aoy; dz=aoz;
    }
    return false;
}

// GJK nearest simplex for a triangle (n==3). A=v[0], B=v[1], C=v[2].
inline bool gjk_triangle(GjkSimplex& s, float& dx, float& dy, float& dz)
{
    float ax=s.v[0].x, ay=s.v[0].y, az=s.v[0].z;
    float bx=s.v[1].x, by=s.v[1].y, bz=s.v[1].z;
    float cx=s.v[2].x, cy=s.v[2].y, cz=s.v[2].z;
    float abx=bx-ax, aby=by-ay, abz=bz-az;
    float acx=cx-ax, acy=cy-ay, acz=cz-az;
    float aox=-ax, aoy=-ay, aoz=-az;
    // Triangle normal
    float nx=aby*acz-abz*acy, ny=abz*acx-abx*acz, nz=abx*acy-aby*acx;

    // Check edge AB (outward perp = AB × N)
    float pABx=aby*nz-abz*ny, pABy=abz*nx-abx*nz, pABz=abx*ny-aby*nx;
    if (pABx*aox+pABy*aoy+pABz*aoz > 0.f) {
        if (abx*aox+aby*aoy+abz*aoz > 0.f) {
            s.v[0]=s.v[0]; s.v[1]=s.v[1]; s.n=2;
            return gjk_line(s, dx, dy, dz);
        } else {
            s.v[0]=s.v[0]; s.n=1; dx=aox; dy=aoy; dz=aoz; return false;
        }
    }
    // Check edge AC (outward perp = N × AC)
    float pACx=ny*acz-nz*acy, pACy=nz*acx-nx*acz, pACz=nx*acy-ny*acx;
    if (pACx*aox+pACy*aoy+pACz*aoz > 0.f) {
        if (acx*aox+acy*aoy+acz*aoz > 0.f) {
            s.v[1]=s.v[2]; s.n=2;
            return gjk_line(s, dx, dy, dz);
        } else {
            s.v[0]=s.v[0]; s.n=1; dx=aox; dy=aoy; dz=aoz; return false;
        }
    }
    // Origin is inside triangle prism
    float dot_n_ao = nx*aox+ny*aoy+nz*aoz;
    if (dot_n_ao >= 0.f) {
        dx=nx; dy=ny; dz=nz;
    } else {
        GjkPt tmp=s.v[1]; s.v[1]=s.v[2]; s.v[2]=tmp; // swap B and C → flip winding
        dx=-nx; dy=-ny; dz=-nz;
    }
    return false;
}

// GJK nearest simplex for a tetrahedron (n==4). Returns true if origin inside.
// A=v[0] (newest), B=v[1], C=v[2], D=v[3].
inline bool gjk_tetrahedron(GjkSimplex& s, float& dx, float& dy, float& dz)
{
    float ax=s.v[0].x, ay=s.v[0].y, az=s.v[0].z;
    float bx=s.v[1].x, by=s.v[1].y, bz=s.v[1].z;
    float cx=s.v[2].x, cy=s.v[2].y, cz=s.v[2].z;
    float dx_=s.v[3].x, dy_=s.v[3].y, dz_=s.v[3].z;
    float abx=bx-ax, aby=by-ay, abz=bz-az;
    float acx=cx-ax, acy=cy-ay, acz=cz-az;
    float adx=dx_-ax, ady=dy_-ay, adz=dz_-az;
    float aox=-ax, aoy=-ay, aoz=-az;

    // Face normals (outward = away from 4th vertex)
    float nABCx=aby*acz-abz*acy, nABCy=abz*acx-abx*acz, nABCz=abx*acy-aby*acx;
    // Ensure outward (away from D)
    if (nABCx*adx+nABCy*ady+nABCz*adz > 0.f) { nABCx=-nABCx; nABCy=-nABCy; nABCz=-nABCz; }

    float nACDx=acy*adz-acz*ady, nACDy=acz*adx-acx*adz, nACDz=acx*ady-acy*adx;
    if (nACDx*abx+nACDy*aby+nACDz*abz > 0.f) { nACDx=-nACDx; nACDy=-nACDy; nACDz=-nACDz; }

    float nADBx=ady*abz-adz*aby, nADBy=adz*abx-adx*abz, nADBz=adx*aby-ady*abx;
    if (nADBx*acx+nADBy*acy+nADBz*acz > 0.f) { nADBx=-nADBx; nADBy=-nADBy; nADBz=-nADBz; }

    bool abcOut = (nABCx*aox+nABCy*aoy+nABCz*aoz > 0.f);
    bool acdOut = (nACDx*aox+nACDy*aoy+nACDz*aoz > 0.f);
    bool adbOut = (nADBx*aox+nADBy*aoy+nADBz*aoz > 0.f);

    if (!abcOut && !acdOut && !adbOut) return true; // origin inside tetrahedron

    if (abcOut) {
        // Collapse to face ABC
        s.v[0]=s.v[0]; s.v[1]=s.v[1]; s.v[2]=s.v[2]; s.n=3;
        return gjk_triangle(s, dx, dy, dz);
    }
    if (acdOut) {
        s.v[0]=s.v[0]; s.v[1]=s.v[2]; s.v[2]=s.v[3]; s.n=3;
        return gjk_triangle(s, dx, dy, dz);
    }
    // adbOut
    s.v[0]=s.v[0]; s.v[1]=s.v[3]; s.v[2]=s.v[1]; s.n=3;
    return gjk_triangle(s, dx, dy, dz);
}

inline bool gjk_nearest(GjkSimplex& s, float& dx, float& dy, float& dz)
{
    switch (s.n) {
        case 2: return gjk_line(s, dx, dy, dz);
        case 3: return gjk_triangle(s, dx, dy, dz);
        case 4: return gjk_tetrahedron(s, dx, dy, dz);
        default: return false;
    }
}

// ── EPA ───────────────────────────────────────────────────────────────────────
// Expands the GJK simplex into a full polytope to find minimum penetration depth.
// Fixed-size arrays: at most EPA_MAX_VERTS vertices, EPA_MAX_FACES triangular faces.

static constexpr int EPA_MAX_VERTS = 64;
static constexpr int EPA_MAX_FACES = 128;
static constexpr int EPA_MAX_ITER  = 32;

struct EpaFace {
    int     v[3];
    float   nx, ny, nz, dist; // unit outward normal, distance from origin to plane
};

struct EpaPolytope {
    GjkPt   verts[EPA_MAX_VERTS];
    EpaFace faces[EPA_MAX_FACES];
    int     n_verts = 0, n_faces = 0;

    bool add_vert(GjkPt p) {
        if (n_verts >= EPA_MAX_VERTS) return false;
        verts[n_verts++] = p; return true;
    }
    bool add_face(int a, int b, int c) {
        if (n_faces >= EPA_MAX_FACES) return false;
        float ax=verts[a].x, ay=verts[a].y, az=verts[a].z;
        float bx=verts[b].x, by=verts[b].y, bz=verts[b].z;
        float cx=verts[c].x, cy=verts[c].y, cz=verts[c].z;
        float abx=bx-ax, aby=by-ay, abz=bz-az;
        float acx=cx-ax, acy=cy-ay, acz=cz-az;
        float nx=aby*acz-abz*acy, ny=abz*acx-abx*acz, nz=abx*acy-aby*acx;
        float len = sycl::sqrt(nx*nx+ny*ny+nz*nz);
        if (len < 1e-10f) return false; // degenerate, skip
        float inv=1.f/len; nx*=inv; ny*=inv; nz*=inv;
        float dist = nx*ax+ny*ay+nz*az;
        if (dist < 0.f) { // normal points inward → flip
            nx=-nx; ny=-ny; nz=-nz; dist=-dist;
            EpaFace& f=faces[n_faces++]; f.v[0]=a; f.v[1]=c; f.v[2]=b;
            f.nx=nx; f.ny=ny; f.nz=nz; f.dist=dist;
        } else {
            EpaFace& f=faces[n_faces++]; f.v[0]=a; f.v[1]=b; f.v[2]=c;
            f.nx=nx; f.ny=ny; f.nz=nz; f.dist=dist;
        }
        return true;
    }
};

// Run EPA. On entry simplex must be a 4-point tetrahedron enclosing the origin.
// Out: normal (from B toward A), depth, contact points on A and B.
inline bool epa(GjkSimplex& gjk_s,
                const ConvexHullView& ca, uint32_t ia,
                float apx,float apy,float apz, float aqw,float aqx,float aqy,float aqz,
                const ConvexHullView& cb, uint32_t ib,
                float bpx,float bpy,float bpz, float bqw,float bqx,float bqy,float bqz,
                float& out_nx, float& out_ny, float& out_nz,
                float& out_depth,
                float& out_cpx, float& out_cpy, float& out_cpz)
{
    EpaPolytope p;
    for (int i=0; i<4; ++i) p.add_vert(gjk_s.v[i]);

    // Build initial 4 faces of the tetrahedron with correct outward normals.
    p.add_face(0,1,2);
    p.add_face(0,2,3);
    p.add_face(0,3,1);
    p.add_face(1,3,2);

    for (int iter=0; iter<EPA_MAX_ITER; ++iter) {
        // Find closest face to origin
        int best_f = 0; float best_d = p.faces[0].dist;
        for (int f=1; f<p.n_faces; ++f)
            if (p.faces[f].dist < best_d) { best_d=p.faces[f].dist; best_f=f; }

        EpaFace& bf = p.faces[best_f];
        // New support point in direction of closest face normal
        GjkPt np = minkdiff(ca,ia, apx,apy,apz, aqw,aqx,aqy,aqz,
                             cb,ib, bpx,bpy,bpz, bqw,bqx,bqy,bqz,
                             bf.nx, bf.ny, bf.nz);

        float new_dist = np.x*bf.nx + np.y*bf.ny + np.z*bf.nz;
        if (new_dist - best_d < 1e-4f) {
            // Converged: closest face is the answer
            out_nx = bf.nx; out_ny = bf.ny; out_nz = bf.nz;
            out_depth = best_d;
            // Interpolate contact point on A and B using barycentric coords
            // (approximate: use average of face vertices' corresponding A/B points)
            int va=bf.v[0], vb=bf.v[1], vc=bf.v[2];
            out_cpx = (p.verts[va].ax + p.verts[vb].ax + p.verts[vc].ax) / 3.f;
            out_cpy = (p.verts[va].ay + p.verts[vb].ay + p.verts[vc].ay) / 3.f;
            out_cpz = (p.verts[va].az + p.verts[vb].az + p.verts[vc].az) / 3.f;
            return true;
        }

        int new_vi = p.n_verts;
        if (!p.add_vert(np)) break; // polytope full

        // Remove all faces visible from new_vi.
        // Collect horizon edges (on boundary between visible and invisible).
        int edge_a[EPA_MAX_FACES], edge_b[EPA_MAX_FACES];
        int n_edges = 0;
        int new_n_faces = 0;

        for (int f=0; f<p.n_faces; ++f) {
            EpaFace& face = p.faces[f];
            float vis = face.nx*np.x + face.ny*np.y + face.nz*np.z;
            if (vis > face.dist) {
                // Face is visible → add its edges to potential horizon
                for (int e=0; e<3; ++e) {
                    int ea=face.v[e], eb=face.v[(e+1)%3];
                    // Check if this edge is already in list (from opposite direction)
                    bool found = false;
                    for (int k=0; k<n_edges; ++k) {
                        if (edge_a[k]==eb && edge_b[k]==ea) {
                            // Shared → remove from horizon (interior edge)
                            edge_a[k]=edge_a[--n_edges]; edge_b[k]=edge_b[n_edges];
                            found=true; break;
                        }
                    }
                    if (!found && n_edges < EPA_MAX_FACES) {
                        edge_a[n_edges]=ea; edge_b[n_edges]=eb; ++n_edges;
                    }
                }
            } else {
                // Face invisible → keep
                p.faces[new_n_faces++] = face;
            }
        }
        p.n_faces = new_n_faces;

        // Add new faces from horizon edges to new vertex
        for (int e=0; e<n_edges; ++e)
            p.add_face(edge_a[e], edge_b[e], new_vi);
    }

    // Fallback: return last best face
    int best_f=0;
    for (int f=1; f<p.n_faces; ++f)
        if (p.faces[f].dist < p.faces[best_f].dist) best_f=f;
    EpaFace& bf=p.faces[best_f];
    out_nx=bf.nx; out_ny=bf.ny; out_nz=bf.nz; out_depth=bf.dist;
    int va=bf.v[0],vb=bf.v[1],vc=bf.v[2];
    out_cpx=(p.verts[va].ax+p.verts[vb].ax+p.verts[vc].ax)/3.f;
    out_cpy=(p.verts[va].ay+p.verts[vb].ay+p.verts[vc].ay)/3.f;
    out_cpz=(p.verts[va].az+p.verts[vb].az+p.verts[vc].az)/3.f;
    return true;
}

// GJK entry point: returns true if overlap (and fills EPA result).
inline bool gjk_epa(
    const ConvexHullView& ca, uint32_t ia,
    float apx,float apy,float apz, float aqw,float aqx,float aqy,float aqz,
    const ConvexHullView& cb, uint32_t ib,
    float bpx,float bpy,float bpz, float bqw,float bqx,float bqy,float bqz,
    float& out_nx,float& out_ny,float& out_nz,
    float& out_depth, float& out_px,float& out_py,float& out_pz)
{
    GjkSimplex s;
    float dx=1.f, dy=0.f, dz=0.f;
    GjkPt sp = minkdiff(ca,ia, apx,apy,apz, aqw,aqx,aqy,aqz,
                         cb,ib, bpx,bpy,bpz, bqw,bqx,bqy,bqz, dx,dy,dz);
    s.push(sp);
    dx=-sp.x; dy=-sp.y; dz=-sp.z;

    for (int iter=0; iter<64; ++iter) {
        float len2=dx*dx+dy*dy+dz*dz;
        if (len2 < 1e-14f) return false;
        sp = minkdiff(ca,ia, apx,apy,apz, aqw,aqx,aqy,aqz,
                      cb,ib, bpx,bpy,bpz, bqw,bqx,bqy,bqz, dx,dy,dz);
        // If the new point is not further than current simplex, shapes don't overlap
        if (sp.x*dx+sp.y*dy+sp.z*dz < 0.f) return false;
        s.push(sp);
        if (gjk_nearest(s, dx, dy, dz)) {
            // Overlap confirmed → run EPA
            return epa(s, ca,ia, apx,apy,apz, aqw,aqx,aqy,aqz,
                          cb,ib, bpx,bpy,bpz, bqw,bqx,bqy,bqz,
                          out_nx,out_ny,out_nz, out_depth, out_px,out_py,out_pz);
        }
    }
    return false;
}

// ── Sphere-vs-triangle (for mesh BVH narrowphase) ────────────────────────────
// Closest point on triangle to sphere center; returns true if within radius.
inline bool sphere_triangle(
    float scx,float scy,float scz, float sr,
    float v0x,float v0y,float v0z,
    float v1x,float v1y,float v1z,
    float v2x,float v2y,float v2z,
    float& px,float& py,float& pz,
    float& nx,float& ny,float& nz,
    float& depth)
{
    // Compute closest point on triangle to sphere center (point-triangle distance)
    float e0x=v1x-v0x, e0y=v1y-v0y, e0z=v1z-v0z;
    float e1x=v2x-v0x, e1y=v2y-v0y, e1z=v2z-v0z;
    float dx=scx-v0x, dy=scy-v0y, dz=scz-v0z;

    float a=e0x*e0x+e0y*e0y+e0z*e0z;
    float b=e0x*e1x+e0y*e1y+e0z*e1z;
    float c=e1x*e1x+e1y*e1y+e1z*e1z;
    float d=e0x*dx+e0y*dy+e0z*dz;
    float e=e1x*dx+e1y*dy+e1z*dz;
    float det=a*c-b*b, s_=b*e-c*d, t_=b*d-a*e;

    if (s_+t_ <= det) {
        if (s_ < 0.f) {
            if (t_ < 0.f) { if (d<0.f) { s_=det<=-d?1.f:(-d/a); t_=0.f; } else { s_=0.f; t_=e>=0.f?0.f:((-e)>=c?1.f:(-e/c)); } }
            else { s_=0.f; t_=e>=0.f?0.f:((-e)>=c?1.f:(-e/c)); }
        } else if (t_ < 0.f) {
            t_=0.f; s_=d>=0.f?0.f:((-d)>=a?1.f:(-d/a));
        }
    } else {
        if (s_ < 0.f) { float tmp=b+d,tmp2=c+e; s_=0.f; t_=tmp2>tmp?(tmp2-tmp)/(a-2.f*b+c):1.f; }
        else if (t_ < 0.f) { float tmp=b+e,tmp2=a+d; t_=0.f; s_=tmp2>tmp?(tmp2-tmp)/(a-2.f*b+c):1.f; }
        else { float inv=1.f/det; s_*=inv; t_*=inv; }
    }
    // Clamp to [0,1]
    s_=s_<0.f?0.f:(s_>1.f?1.f:s_);
    t_=t_<0.f?0.f:(t_>1.f?1.f:t_);

    float cpx=v0x+s_*e0x+t_*e1x;
    float cpy=v0y+s_*e0y+t_*e1y;
    float cpz=v0z+s_*e0z+t_*e1z;
    float diffx=scx-cpx, diffy=scy-cpy, diffz=scz-cpz;
    float dist2=diffx*diffx+diffy*diffy+diffz*diffz;
    if (dist2 >= sr*sr) return false;

    float dist=sycl::sqrt(dist2);
    if (dist < 1e-8f) {
        // Compute triangle normal
        float tnx=e0y*e1z-e0z*e1y, tny=e0z*e1x-e0x*e1z, tnz=e0x*e1y-e0y*e1x;
        float len=sycl::sqrt(tnx*tnx+tny*tny+tnz*tnz);
        if (len<1e-8f) { nx=0;ny=1;nz=0; } else { nx=tnx/len;ny=tny/len;nz=tnz/len; }
        // Orient toward sphere center
        if (nx*(scx-v0x)+ny*(scy-v0y)+nz*(scz-v0z) < 0) { nx=-nx;ny=-ny;nz=-nz; }
        depth=sr;
    } else {
        float inv=1.f/dist;
        nx=diffx*inv; ny=diffy*inv; nz=diffz*inv;
        depth=sr-dist;
    }
    px=cpx; py=cpy; pz=cpz;
    return true;
}

// ── Box-vs-triangle SAT (13 axes) ────────────────────────────────────────────
// Returns penetration depth (> 0 means overlap) and fills normal.
// Box is centered at origin in its own local frame (extents hx,hy,hz).
inline bool box_triangle_local(
    float hx, float hy, float hz,
    // triangle vertices in box local frame
    float ax,float ay,float az,
    float bx,float by,float bz,
    float cx,float cy,float cz,
    float& nx,float& ny,float& nz,
    float& depth)
{
    // Minimum penetration tracking
    float best = 1e30f;
    nx=0; ny=1; nz=0;

    auto test_axis = [&](float lx,float ly,float lz) -> bool {
        float len2=lx*lx+ly*ly+lz*lz;
        if (len2 < 1e-10f) return true; // degenerate, skip
        float inv=sycl::rsqrt(len2);
        lx*=inv; ly*=inv; lz*=inv;
        float rA = hx*sycl::fabs(lx)+hy*sycl::fabs(ly)+hz*sycl::fabs(lz);
        float dA=ax*lx+ay*ly+az*lz;
        float dB=bx*lx+by*ly+bz*lz;
        float dC=cx*lx+cy*ly+cz*lz;
        float tmin=sycl::fmin(dA,sycl::fmin(dB,dC));
        float tmax=sycl::fmax(dA,sycl::fmax(dB,dC));
        float ov=sycl::fmin(rA-tmin, tmax+rA);
        if (ov <= 0.f) return false;
        if (ov < best) { best=ov; nx=lx;ny=ly;nz=lz;
            // Orient toward triangle centroid
            float ctr=(dA+dB+dC)/3.f;
            if (ctr < 0) { nx=-nx;ny=-ny;nz=-nz; }
        }
        return true;
    };

    // 3 box face normals
    if (!test_axis(1,0,0)) return false;
    if (!test_axis(0,1,0)) return false;
    if (!test_axis(0,0,1)) return false;
    // Triangle normal
    float enx=bx-ax,eny=by-ay,enz=bz-az;
    float emx=cx-ax,emy=cy-ay,emz=cz-az;
    if (!test_axis(eny*emz-enz*emy, enz*emx-enx*emz, enx*emy-eny*emx)) return false;
    // 9 edge-edge cross products: box edges (3) × triangle edges (3)
    float tedges[3][3] = {{bx-ax,by-ay,bz-az},{cx-bx,cy-by,cz-bz},{ax-cx,ay-cy,az-cz}};
    float box_axes[3][3] = {{1,0,0},{0,1,0},{0,0,1}};
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) {
        float lx=box_axes[i][1]*tedges[j][2]-box_axes[i][2]*tedges[j][1];
        float ly=box_axes[i][2]*tedges[j][0]-box_axes[i][0]*tedges[j][2];
        float lz=box_axes[i][0]*tedges[j][1]-box_axes[i][1]*tedges[j][0];
        if (!test_axis(lx,ly,lz)) return false;
    }
    depth = best;
    return true;
}

// ── AABB vs AABB (for BVH traversal) ─────────────────────────────────────────
inline bool aabb_aabb(
    float amin_x,float amin_y,float amin_z,
    float amax_x,float amax_y,float amax_z,
    float bmin_x,float bmin_y,float bmin_z,
    float bmax_x,float bmax_y,float bmax_z)
{
    return !(amax_x < bmin_x || bmax_x < amin_x ||
             amax_y < bmin_y || bmax_y < amin_y ||
             amax_z < bmin_z || bmax_z < amin_z);
}

// ── BVH traversal for one dynamic body vs. one static mesh ───────────────────
// Uses a fixed-size stack. Returns number of contacts emitted.
static constexpr int BVH_STACK_SIZE = 64;

inline void traverse_sphere_mesh(
    PairSink& sk, uint32_t ia, uint32_t ib,
    // Sphere (world space)
    float scx,float scy,float scz, float sr,
    // BVH data (nodes in local space of ib; ib is static so local=world here)
    const float* vtx_x,const float* vtx_y,const float* vtx_z,
    const uint32_t* tri_a,const uint32_t* tri_b,const uint32_t* tri_c,
    const BvhNode* nodes, uint32_t root)
{
    int stack[BVH_STACK_SIZE]; int top=0;
    stack[top++]=static_cast<int>(root);
    while (top>0) {
        int ni=stack[--top];
        const BvhNode& nd=nodes[ni];
        // Sphere-AABB overlap check
        float cx=scx<nd.min_x?nd.min_x:(scx>nd.max_x?nd.max_x:scx);
        float cy=scy<nd.min_y?nd.min_y:(scy>nd.max_y?nd.max_y:scy);
        float cz=scz<nd.min_z?nd.min_z:(scz>nd.max_z?nd.max_z:scz);
        float dx=scx-cx,dy=scy-cy,dz=scz-cz;
        if (dx*dx+dy*dy+dz*dz >= sr*sr) continue;
        if (nd.left < 0) {
            // Leaf
            uint32_t t=static_cast<uint32_t>(nd.tri_idx);
            float px,py,pz,nx,ny,nz,depth;
            if (sphere_triangle(scx,scy,scz,sr,
                                vtx_x[tri_a[t]],vtx_y[tri_a[t]],vtx_z[tri_a[t]],
                                vtx_x[tri_b[t]],vtx_y[tri_b[t]],vtx_z[tri_b[t]],
                                vtx_x[tri_c[t]],vtx_y[tri_c[t]],vtx_z[tri_c[t]],
                                px,py,pz,nx,ny,nz,depth))
                emit(sk,ia,ib,px,py,pz,nx,ny,nz,depth);
        } else {
            if (top+2 < BVH_STACK_SIZE) {
                stack[top++]=nd.left; stack[top++]=nd.right;
            }
        }
    }
}

inline void traverse_box_mesh(
    PairSink& sk, uint32_t ia, uint32_t ib,
    // Box world-space center + rotation matrix + half-extents
    float bcx,float bcy,float bcz,
    const float Re[3][3], float hx,float hy,float hz,
    const float* vtx_x,const float* vtx_y,const float* vtx_z,
    const uint32_t* tri_a,const uint32_t* tri_b,const uint32_t* tri_c,
    const BvhNode* nodes, uint32_t root)
{
    // World-space AABB of the box (for BVH overlap)
    float bmin_x = bcx-(hx+hy+hz), bmin_y = bcy-(hx+hy+hz), bmin_z = bcz-(hx+hy+hz);
    float bmax_x = bcx+(hx+hy+hz), bmax_y = bcy+(hx+hy+hz), bmax_z = bcz+(hx+hy+hz);

    int stack[BVH_STACK_SIZE]; int top=0;
    stack[top++]=static_cast<int>(root);
    while (top>0) {
        int ni=stack[--top];
        const BvhNode& nd=nodes[ni];
        if (!aabb_aabb(bmin_x,bmin_y,bmin_z,bmax_x,bmax_y,bmax_z,
                       nd.min_x,nd.min_y,nd.min_z,nd.max_x,nd.max_y,nd.max_z)) continue;
        if (nd.left < 0) {
            // Leaf: transform triangle to box-local space and do SAT
            uint32_t t=static_cast<uint32_t>(nd.tri_idx);
            // World-space triangle vertices
            float wx0=vtx_x[tri_a[t]],wy0=vtx_y[tri_a[t]],wz0=vtx_z[tri_a[t]];
            float wx1=vtx_x[tri_b[t]],wy1=vtx_y[tri_b[t]],wz1=vtx_z[tri_b[t]];
            float wx2=vtx_x[tri_c[t]],wy2=vtx_y[tri_c[t]],wz2=vtx_z[tri_c[t]];
            // Transform to box-local frame (R^T * (v - center))
            auto to_local = [&](float wx,float wy,float wz, float& lx,float& ly,float& lz){
                float dx=wx-bcx,dy=wy-bcy,dz=wz-bcz;
                lx=dx*Re[0][0]+dy*Re[0][1]+dz*Re[0][2];
                ly=dx*Re[1][0]+dy*Re[1][1]+dz*Re[1][2];
                lz=dx*Re[2][0]+dy*Re[2][1]+dz*Re[2][2];
            };
            float lax,lay,laz, lbx,lby,lbz, lcx,lcy,lcz;
            to_local(wx0,wy0,wz0, lax,lay,laz);
            to_local(wx1,wy1,wz1, lbx,lby,lbz);
            to_local(wx2,wy2,wz2, lcx,lcy,lcz);
            float nx_l,ny_l,nz_l,depth;
            if (box_triangle_local(hx,hy,hz, lax,lay,laz, lbx,lby,lbz, lcx,lcy,lcz,
                                   nx_l,ny_l,nz_l,depth)) {
                // Transform normal back to world
                float nx_w=nx_l*Re[0][0]+ny_l*Re[1][0]+nz_l*Re[2][0];
                float ny_w=nx_l*Re[0][1]+ny_l*Re[1][1]+nz_l*Re[2][1];
                float nz_w=nx_l*Re[0][2]+ny_l*Re[1][2]+nz_l*Re[2][2];
                // Contact point: midpoint of triangle centroid and box surface
                float tcx=(wx0+wx1+wx2)/3.f, tcy=(wy0+wy1+wy2)/3.f, tcz=(wz0+wz1+wz2)/3.f;
                emit(sk,ia,ib,tcx,tcy,tcz,nx_w,ny_w,nz_w,depth);
            }
        } else {
            if (top+2 < BVH_STACK_SIZE) {
                stack[top++]=nd.left; stack[top++]=nd.right;
            }
        }
    }
}

// ── make_axes helper ─────────────────────────────────────────────────────────
inline void make_axes(float qw,float qx,float qy,float qz, float Ae[3][3]) {
    Ae[0][0]=1.f-2.f*(qy*qy+qz*qz); Ae[0][1]=2.f*(qx*qy+qz*qw); Ae[0][2]=2.f*(qx*qz-qy*qw);
    Ae[1][0]=2.f*(qx*qy-qz*qw);     Ae[1][1]=1.f-2.f*(qx*qx+qz*qz); Ae[1][2]=2.f*(qy*qz+qx*qw);
    Ae[2][0]=2.f*(qx*qz+qy*qw);     Ae[2][1]=2.f*(qy*qz-qx*qw); Ae[2][2]=1.f-2.f*(qx*qx+qy*qy);
}

// ── Per-pair dispatch (shared by both run() overloads) ────────────────────────
static inline void process_pair(
    uint32_t idx, const ContactPair* d_pairs,
    const BodyView& bv, const ShapeView& sv,
    ConvexHullView hv, MeshBvhCatalogView mv,
    PairSink& sk)
{
    ContactPair pair = d_pairs[idx];
    uint32_t ia = pair.a, ib = pair.b;

    uint32_t sha = bv.shape[ia], shb = bv.shape[ib];
    uint32_t ta  = sv.type[sha], tb  = sv.type[shb];

    constexpr uint32_t kBox    = static_cast<uint32_t>(ShapeType::Box);
    constexpr uint32_t kSphere = static_cast<uint32_t>(ShapeType::Sphere);
    constexpr uint32_t kHull   = static_cast<uint32_t>(ShapeType::ConvexHull);
    constexpr uint32_t kMesh   = static_cast<uint32_t>(ShapeType::TriangleMesh);

    // Swap so that TriangleMesh is always body B (static)
    if (ta == kMesh) {
        uint32_t tmp=ia; ia=ib; ib=tmp;
        uint32_t tmps=sha; sha=shb; shb=tmps;
        uint32_t tmpt=ta; ta=tb; tb=tmpt;
    }

    float px, py, pz, nx, ny, nz, depth;

    if (ta == kSphere && tb == kSphere) {
        if (!sphere_sphere(bv.pos_x[ia], bv.pos_y[ia], bv.pos_z[ia], sv.half_x[sha],
                           bv.pos_x[ib], bv.pos_y[ib], bv.pos_z[ib], sv.half_x[shb],
                           px, py, pz, nx, ny, nz, depth)) return;
        emit(sk, ia, ib, px, py, pz, nx, ny, nz, depth);
        return;
    }

    if (ta == kSphere && tb == kBox) {
        float Be[3][3]; float Bh[3] = {sv.half_x[shb], sv.half_y[shb], sv.half_z[shb]};
        make_axes(bv.rot_w[ib], bv.rot_x[ib], bv.rot_y[ib], bv.rot_z[ib], Be);
        if (!sphere_box(bv.pos_x[ia], bv.pos_y[ia], bv.pos_z[ia], sv.half_x[sha],
                        bv.pos_x[ib], bv.pos_y[ib], bv.pos_z[ib],
                        Be[0][0], Be[0][1], Be[0][2],
                        Be[1][0], Be[1][1], Be[1][2],
                        Be[2][0], Be[2][1], Be[2][2],
                        Bh[0], Bh[1], Bh[2], px, py, pz, nx, ny, nz, depth)) return;
        nx = -nx; ny = -ny; nz = -nz;
        emit(sk, ia, ib, px, py, pz, nx, ny, nz, depth);
        return;
    }

    if (ta == kBox && tb == kSphere) {
        float Ae[3][3]; float Ah[3] = {sv.half_x[sha], sv.half_y[sha], sv.half_z[sha]};
        make_axes(bv.rot_w[ia], bv.rot_x[ia], bv.rot_y[ia], bv.rot_z[ia], Ae);
        if (!sphere_box(bv.pos_x[ib], bv.pos_y[ib], bv.pos_z[ib], sv.half_x[shb],
                        bv.pos_x[ia], bv.pos_y[ia], bv.pos_z[ia],
                        Ae[0][0], Ae[0][1], Ae[0][2],
                        Ae[1][0], Ae[1][1], Ae[1][2],
                        Ae[2][0], Ae[2][1], Ae[2][2],
                        Ah[0], Ah[1], Ah[2], px, py, pz, nx, ny, nz, depth)) return;
        emit(sk, ia, ib, px, py, pz, nx, ny, nz, depth);
        return;
    }

    if (ta == kBox && tb == kBox) {
        float Ae[3][3]; float Ah[3] = {sv.half_x[sha], sv.half_y[sha], sv.half_z[sha]};
        float Be_m[3][3]; float Bh[3] = {sv.half_x[shb], sv.half_y[shb], sv.half_z[shb]};
        make_axes(bv.rot_w[ia], bv.rot_x[ia], bv.rot_y[ia], bv.rot_z[ia], Ae);
        make_axes(bv.rot_w[ib], bv.rot_x[ib], bv.rot_y[ib], bv.rot_z[ib], Be_m);
        float opx[4], opy[4], opz[4], od[4];
        float onx, ony, onz;
        int nc = box_box(bv.pos_x[ia], bv.pos_y[ia], bv.pos_z[ia], Ae, Ah,
                         bv.pos_x[ib], bv.pos_y[ib], bv.pos_z[ib], Be_m, Bh,
                         opx, opy, opz, onx, ony, onz, od);
        // Reduce to centroid: sequential application of 4 angular corrections
        // causes non-commutative quaternion drift that destabilizes the simulation.
        if (nc > 0) {
            float cx = 0, cy = 0, cz = 0, cd = 0;
            for (int k = 0; k < nc; ++k) { cx+=opx[k]; cy+=opy[k]; cz+=opz[k]; cd+=od[k]; }
            float inv = 1.f / nc;
            emit(sk, ia, ib, cx*inv, cy*inv, cz*inv, onx, ony, onz, cd*inv);
        }
        return;
    }

    if (ta == kHull || tb == kHull) {
        if (hv.n_hulls == 0) return;
        if (ta == kHull && tb == kHull) {
            uint32_t hull_a = sv.ext_id[sha], hull_b = sv.ext_id[shb];
            float onx,ony,onz,odepth,opx,opy,opz;
            if (gjk_epa(hv, hull_a,
                         bv.pos_x[ia],bv.pos_y[ia],bv.pos_z[ia],
                         bv.rot_w[ia],bv.rot_x[ia],bv.rot_y[ia],bv.rot_z[ia],
                         hv, hull_b,
                         bv.pos_x[ib],bv.pos_y[ib],bv.pos_z[ib],
                         bv.rot_w[ib],bv.rot_x[ib],bv.rot_y[ib],bv.rot_z[ib],
                         onx,ony,onz,odepth,opx,opy,opz))
                emit(sk,ia,ib,opx,opy,opz,onx,ony,onz,odepth);
        }
        return;
    }

    if (tb == kMesh) {
        if (mv.n_meshes == 0) return;
        uint32_t mesh_id = sv.ext_id[shb];
        if (mesh_id >= mv.n_meshes) return;

        uint32_t vtx_off  = mv.vtx_offset[mesh_id];
        uint32_t tri_off  = mv.tri_offset[mesh_id];
        uint32_t root     = mv.root_node[mesh_id];
        const float*    vx = mv.vtx_x + vtx_off;
        const float*    vy = mv.vtx_y + vtx_off;
        const float*    vz = mv.vtx_z + vtx_off;
        const uint32_t* ta_ = mv.tri_a + tri_off;
        const uint32_t* tb_ = mv.tri_b + tri_off;
        const uint32_t* tc_ = mv.tri_c + tri_off;
        const BvhNode*  nodes = mv.nodes;

        if (ta == kSphere) {
            traverse_sphere_mesh(sk, ia, ib,
                bv.pos_x[ia],bv.pos_y[ia],bv.pos_z[ia], sv.half_x[sha],
                vx,vy,vz, ta_,tb_,tc_, nodes, root);
        } else if (ta == kBox) {
            float Re[3][3]; float ah[3]={sv.half_x[sha],sv.half_y[sha],sv.half_z[sha]};
            make_axes(bv.rot_w[ia],bv.rot_x[ia],bv.rot_y[ia],bv.rot_z[ia], Re);
            traverse_box_mesh(sk,ia,ib,
                bv.pos_x[ia],bv.pos_y[ia],bv.pos_z[ia], Re, ah[0],ah[1],ah[2],
                vx,vy,vz, ta_,tb_,tc_, nodes, root);
        }
    }
}

} // anonymous namespace

// ── Narrowphase constructor ───────────────────────────────────────────────────

Narrowphase::Narrowphase(Stream& s, uint32_t max_contacts)
    : store_(s, max_contacts)
    , scratch_(s, max_contacts)
    , keys_(s, next_pow2(max_contacts))
    , perm_(s, next_pow2(max_contacts))
    , cap_(max_contacts)
    , pad_cap_(next_pow2(max_contacts))
{}

// ── Narrowphase::run (CPU pair count) ────────────────────────────────────────
//
// One work-item per candidate pair runs the contact tests in parallel, appending
// to scratch_ (unordered) and tagging each contact with a (pair_idx, sub_idx) key.
// The contacts are then sorted by that key and gathered into store_, reproducing
// the exact pair-by-pair *order* a sequential pass would produce. The order and
// the set of contacts are therefore identical to the old single-work-item path,
// so the solver and sensors see the same input and determinism is preserved.
//
// Note: this changes the contact *values* by at most 1 ULP on the CUDA backend —
// the JIT (ptxas) contracts a*b+c into FMA differently when the shared geometry
// code is inlined into a wide parallel_for than into the old single-thread loop.
// That is the same FMA-contraction effect documented in known_issues Issue 1; it
// is deterministic per build. The CPU/OpenMP path is bit-identical to before.

void Narrowphase::run(Stream& s,
                      const ContactPair* d_pairs, uint32_t n_pairs,
                      const BodyView& bodies, const ShapeView& shapes,
                      ConvexHullView hulls,
                      MeshBvhCatalogView meshes)
{
    store_.reset(s);
    scratch_.reset(s);
    last_count_ = 0;
    if (n_pairs == 0) return;

    auto& q = s.queue();
    ContactView scv = scratch_.view();
    uint64_t*   keys = keys_.data();
    uint32_t*   perm = perm_.data();
    const BodyView  bv = bodies;
    const ShapeView sv = shapes;
    const ConvexHullView     hv = hulls;
    const MeshBvhCatalogView mv = meshes;

    // 1. Parallel contact generation → scratch_ + per-contact keys.
    parallel_for(s, n_pairs, [=](size_t i) {
        PairSink sk{scv, keys, static_cast<uint32_t>(i), 0u};
        process_pair(static_cast<uint32_t>(i), d_pairs, bv, sv, hv, mv, sk);
    });

    // 2. How many contacts were produced? (one sync; the broadphase already syncs
    //    once per frame, so this adds a single extra round-trip.)
    uint32_t nc = scratch_.download_count(s);
    last_count_ = nc;                 // true count (may exceed cap_ → overflow drop)
    if (nc > cap_) nc = cap_;
    if (nc == 0) return;

    // 3. Sort the contacts by key. Pad to a power of two with UINT64_MAX keys so the
    //    padding sorts to the end; perm starts as identity.
    uint32_t n_sort = next_pow2(nc);
    q.memset(keys + nc, 0xFF, static_cast<size_t>(n_sort - nc) * sizeof(uint64_t));
    parallel_for(s, n_sort, [perm](size_t i) { perm[i] = static_cast<uint32_t>(i); });
    sort_by_key(s, keys, perm, static_cast<size_t>(n_sort));

    // 4. Gather scratch_[perm[k]] → store_[k] for k in [0, nc); set the count.
    ContactView mcv = store_.view();
    parallel_for(s, nc, [=](size_t k) {
        uint32_t src = perm[k];
        mcv.body_a[k] = scv.body_a[src];  mcv.body_b[k] = scv.body_b[src];
        mcv.pos_x[k]  = scv.pos_x[src];   mcv.pos_y[k]  = scv.pos_y[src];  mcv.pos_z[k]  = scv.pos_z[src];
        mcv.norm_x[k] = scv.norm_x[src];  mcv.norm_y[k] = scv.norm_y[src]; mcv.norm_z[k] = scv.norm_z[src];
        mcv.depth[k]  = scv.depth[src];
    });
    parallel_for(s, 1, [mcv, nc](size_t) { *mcv.n = nc; });
}

// ── Narrowphase::run (device pair count) ──────────────────────────────────────
//
// The pair count must be known on the host to size the parallel dispatch and the
// contact sort, so this variant downloads it and delegates to the CPU-count path.
// (No current caller relies on it staying sync-free.)

void Narrowphase::run(Stream& s,
                      const ContactPair* d_pairs, const uint32_t* d_n_pairs,
                      const BodyView& bodies, const ShapeView& shapes,
                      ConvexHullView hulls,
                      MeshBvhCatalogView meshes)
{
    uint32_t n_pairs;
    s.queue().memcpy(&n_pairs, d_n_pairs, sizeof(n_pairs)).wait();
    run(s, d_pairs, n_pairs, bodies, shapes, hulls, meshes);
}

} // namespace dyphur
