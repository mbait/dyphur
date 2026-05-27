#include <core/xpbd_solver.hpp>
#include <compute/kernel.hpp>
#include <sycl/sycl.hpp>

namespace dyphur {
namespace {

// ── Math helpers (device-safe, no Eigen) ─────────────────────────────────────

// World-space inverse inertia applied to v: I_world_inv * v.
// BodyView is taken by const-ref; we only read from it here.
inline void world_inv_inertia(const BodyView& bv, uint32_t bi,
                               float vx, float vy, float vz,
                               float& ox, float& oy, float& oz) {
    float qw = bv.rot_w[bi], qx = bv.rot_x[bi];
    float qy = bv.rot_y[bi], qz = bv.rot_z[bi];
    float R00=1.f-2.f*(qy*qy+qz*qz), R01=2.f*(qx*qy+qz*qw), R02=2.f*(qx*qz-qy*qw);
    float R10=2.f*(qx*qy-qz*qw),     R11=1.f-2.f*(qx*qx+qz*qz), R12=2.f*(qy*qz+qx*qw);
    float R20=2.f*(qx*qz+qy*qw),     R21=2.f*(qy*qz-qx*qw),     R22=1.f-2.f*(qx*qx+qy*qy);
    // body frame: column indexing
    float lx = R00*vx + R10*vy + R20*vz;
    float ly = R01*vx + R11*vy + R21*vz;
    float lz = R02*vx + R12*vy + R22*vz;
    float ilx = bv.iI_xx[bi]*lx + bv.iI_xy[bi]*ly + bv.iI_xz[bi]*lz;
    float ily = bv.iI_xy[bi]*lx + bv.iI_yy[bi]*ly + bv.iI_yz[bi]*lz;
    float ilz = bv.iI_xz[bi]*lx + bv.iI_yz[bi]*ly + bv.iI_zz[bi]*lz;
    // world frame: row indexing
    ox = R00*ilx + R01*ily + R02*ilz;
    oy = R10*ilx + R11*ily + R12*ilz;
    oz = R20*ilx + R21*ily + R22*ilz;
}

// Rotate a body-local vector to world frame using body bi's quaternion.
// Uses column indexing on the transposed rotation matrix.
inline void qrot_to_world(const BodyView& bv, uint32_t bi,
                           float vx, float vy, float vz,
                           float& ox, float& oy, float& oz) {
    float qw = bv.rot_w[bi], qx = bv.rot_x[bi];
    float qy = bv.rot_y[bi], qz = bv.rot_z[bi];
    float R00=1.f-2.f*(qy*qy+qz*qz), R01=2.f*(qx*qy+qz*qw), R02=2.f*(qx*qz-qy*qw);
    float R10=2.f*(qx*qy-qz*qw),     R11=1.f-2.f*(qx*qx+qz*qz), R12=2.f*(qy*qz+qx*qw);
    float R20=2.f*(qx*qz+qy*qw),     R21=2.f*(qy*qz-qx*qw),     R22=1.f-2.f*(qx*qx+qy*qy);
    ox = R00*vx + R10*vy + R20*vz;
    oy = R01*vx + R11*vy + R21*vz;
    oz = R02*vx + R12*vy + R22*vz;
}

// Apply quaternion update: q += 0.5 * [0, omega] * q  (omega in world frame).
// Const-ref is intentional: we modify through the raw pointers, not the struct.
inline void apply_ang_delta(const BodyView& bv, uint32_t bi,
                             float ax, float ay, float az) {
    float qw = bv.rot_w[bi], qx = bv.rot_x[bi];
    float qy = bv.rot_y[bi], qz = bv.rot_z[bi];
    bv.rot_w[bi] += 0.5f*(-ax*qx - ay*qy - az*qz);
    bv.rot_x[bi] += 0.5f*( ax*qw + ay*qz - az*qy);
    bv.rot_y[bi] += 0.5f*(-ax*qz + ay*qw + az*qx);
    bv.rot_z[bi] += 0.5f*( ax*qy - ay*qx + az*qw);
}

// Generalised inverse mass for a constraint along direction (nx,ny,nz)
// acting through lever arm (rx,ry,rz) on body bi. Read-only access to bv.
inline float gen_inv_mass(const BodyView& bv, uint32_t bi,
                           float rx, float ry, float rz,
                           float nx, float ny, float nz) {
    float rxnx = ry*nz - rz*ny, rxny = rz*nx - rx*nz, rxnz = rx*ny - ry*nx;
    float ix, iy, iz;
    world_inv_inertia(bv, bi, rxnx, rxny, rxnz, ix, iy, iz);
    return bv.inv_mass[bi] + (rxnx*ix + rxny*iy + rxnz*iz);
}

// Apply a positional impulse (sign * Δλ * n) to body bi, updating position
// and quaternion.  sign = +1 for the "pushed away" body, -1 for the other.
inline void apply_pos_impulse(const BodyView& bv, uint32_t bi, float sign,
                               float delta_lambda,
                               float nx, float ny, float nz,
                               float rx, float ry, float rz) {
    float im = bv.inv_mass[bi];
    bv.pos_x[bi] += sign * im * delta_lambda * nx;
    bv.pos_y[bi] += sign * im * delta_lambda * ny;
    bv.pos_z[bi] += sign * im * delta_lambda * nz;

    // Angular: ω = I_w^-1 * (r × (sign * Δλ * n))
    float rx_n_x = ry*nz - rz*ny, rx_n_y = rz*nx - rx*nz, rx_n_z = rx*ny - ry*nx;
    float ix, iy, iz;
    world_inv_inertia(bv, bi, rx_n_x, rx_n_y, rx_n_z, ix, iy, iz);
    apply_ang_delta(bv, bi, sign * delta_lambda * ix,
                            sign * delta_lambda * iy,
                            sign * delta_lambda * iz);
}

// ── 3D positional constraint between two bodies ──────────────────────────────
// Anchors are given in world space. Returns false if violation is below eps.
inline bool solve_pos_constraint(const BodyView& bv,
                                  uint32_t ia, uint32_t ib,
                                  float rax, float ray, float raz,  // world-frame lever arm a
                                  float rbx, float rby, float rbz,
                                  float alpha_h2,                   // compliance / h²
                                  float eps = 1e-7f) {
    float cx = (bv.pos_x[ia] + rax) - (bv.pos_x[ib] + rbx);
    float cy = (bv.pos_y[ia] + ray) - (bv.pos_y[ib] + rby);
    float cz = (bv.pos_z[ia] + raz) - (bv.pos_z[ib] + rbz);
    float c2 = cx*cx + cy*cy + cz*cz;
    if (c2 < eps*eps) return false;

    float c_len = sycl::sqrt(c2);
    float nx = cx / c_len, ny = cy / c_len, nz = cz / c_len;

    float wa = gen_inv_mass(bv, ia, rax, ray, raz, nx, ny, nz);
    float wb = gen_inv_mass(bv, ib, rbx, rby, rbz, nx, ny, nz);
    float w  = wa + wb + alpha_h2;
    if (w < 1e-10f) return false;

    float dl = -c_len / w;
    apply_pos_impulse(bv, ia, +1.f, dl, nx, ny, nz, rax, ray, raz);
    apply_pos_impulse(bv, ib, -1.f, dl, nx, ny, nz, rbx, rby, rbz);
    return true;
}

// ── Transverse positional constraint (prismatic sliding) ─────────────────────
// Projects violation onto the plane perpendicular to axis_world and corrects
// only the transverse component, leaving the prismatic DOF free.
inline void solve_transverse_pos(const BodyView& bv,
                                  uint32_t ia, uint32_t ib,
                                  float rax, float ray, float raz,
                                  float rbx, float rby, float rbz,
                                  float axw, float ayw, float azw,  // joint axis, world frame
                                  float alpha_h2) {
    float cx = (bv.pos_x[ia] + rax) - (bv.pos_x[ib] + rbx);
    float cy = (bv.pos_y[ia] + ray) - (bv.pos_y[ib] + rby);
    float cz = (bv.pos_z[ia] + raz) - (bv.pos_z[ib] + rbz);
    // Remove component along axis
    float proj = cx*axw + cy*ayw + cz*azw;
    float tx = cx - proj*axw, ty = cy - proj*ayw, tz = cz - proj*azw;
    float t2 = tx*tx + ty*ty + tz*tz;
    if (t2 < 1e-14f) return;

    float t_len = sycl::sqrt(t2);
    float nx = tx / t_len, ny = ty / t_len, nz = tz / t_len;

    float wa = gen_inv_mass(bv, ia, rax, ray, raz, nx, ny, nz);
    float wb = gen_inv_mass(bv, ib, rbx, rby, rbz, nx, ny, nz);
    float w  = wa + wb + alpha_h2;
    if (w < 1e-10f) return;

    float dl = -t_len / w;
    apply_pos_impulse(bv, ia, +1.f, dl, nx, ny, nz, rax, ray, raz);
    apply_pos_impulse(bv, ib, -1.f, dl, nx, ny, nz, rbx, rby, rbz);
}

// ── Angular constraint between two bodies along a world-space direction ───────
// C_ang is the violation vector (world frame); its magnitude is the error angle.
inline void solve_ang_constraint(const BodyView& bv,
                                  uint32_t ia, uint32_t ib,
                                  float cx, float cy, float cz,
                                  float alpha_h2) {
    float c2 = cx*cx + cy*cy + cz*cz;
    if (c2 < 1e-14f) return;

    float c_len = sycl::sqrt(c2);
    float nx = cx / c_len, ny = cy / c_len, nz = cz / c_len;

    float ix, iy, iz, jx, jy, jz;
    world_inv_inertia(bv, ia, nx, ny, nz, ix, iy, iz);
    world_inv_inertia(bv, ib, nx, ny, nz, jx, jy, jz);
    float wa = nx*ix + ny*iy + nz*iz;
    float wb = nx*jx + ny*jy + nz*jz;
    float w  = wa + wb + alpha_h2;
    if (w < 1e-10f) return;

    float dl = -c_len / w;
    apply_ang_delta(bv, ia, -dl*ix, -dl*iy, -dl*iz);
    apply_ang_delta(bv, ib,  dl*jx,  dl*jy,  dl*jz);
}

// ── Full quaternion angular constraint (Fixed joint) ─────────────────────────
// q_rel = conj(q_ia) * q_ib; violation = 2 * q_rel.xyz rotated to world frame.
inline void solve_fixed_ang(const BodyView& bv, uint32_t ia, uint32_t ib,
                              float alpha_h2) {
    // q_rel = conj(q_a) * q_b
    float aw = bv.rot_w[ia], ax = bv.rot_x[ia], ay = bv.rot_y[ia], az = bv.rot_z[ia];
    float bw = bv.rot_w[ib], bx = bv.rot_x[ib], by = bv.rot_y[ib], bz = bv.rot_z[ib];
    // conj(q_a) * q_b
    float rw =  aw*bw + ax*bx + ay*by + az*bz;
    float rx = -ax*bw + aw*bx - az*by + ay*bz;
    float ry = -ay*bw + az*bx + aw*by - ax*bz;
    float rz = -az*bw - ay*bx + ax*by + aw*bz;
    // Ensure shortest path
    if (rw < 0.f) { rx = -rx; ry = -ry; rz = -rz; }
    // Violation in parent (a) world frame: C = 2 * rotate(q_a, (rx,ry,rz))
    float cx, cy, cz;
    qrot_to_world(bv, ia, rx, ry, rz, cx, cy, cz);
    cx *= 2.f; cy *= 2.f; cz *= 2.f;
    solve_ang_constraint(bv, ia, ib, cx, cy, cz, alpha_h2);
}

// ── Revolute axis-alignment constraint ────────────────────────────────────────
// Aligns axis_ib (child axis in world frame) to axis_ia (parent axis in world).
// C = axis_ia × axis_ib  (zero when aligned; magnitude = sin of misalignment).
inline void solve_revolute_ang(const BodyView& bv, uint32_t ia, uint32_t ib,
                                float ax_local, float ay_local, float az_local,
                                float alpha_h2) {
    float ax_ia, ay_ia, az_ia;
    qrot_to_world(bv, ia, ax_local, ay_local, az_local, ax_ia, ay_ia, az_ia);
    float ax_ib, ay_ib, az_ib;
    qrot_to_world(bv, ib, ax_local, ay_local, az_local, ax_ib, ay_ib, az_ib);
    // Ensure axes in same hemisphere to avoid 180° flip
    if (ax_ia*ax_ib + ay_ia*ay_ib + az_ia*az_ib < 0.f) {
        ax_ib = -ax_ib; ay_ib = -ay_ib; az_ib = -az_ib;
    }
    float cx = ay_ia*az_ib - az_ia*ay_ib;
    float cy = az_ia*ax_ib - ax_ia*az_ib;
    float cz = ax_ia*ay_ib - ay_ia*ax_ib;
    solve_ang_constraint(bv, ia, ib, cx, cy, cz, alpha_h2);
}

// ── Revolute / Prismatic limit constraint ─────────────────────────────────────
// Returns current angle (rad) for revolute, current offset (m) for prismatic.
// Uses a stable reference direction perpendicular to the joint axis.
inline float joint_current_value(const BodyView& bv,
                                  uint32_t ia, uint32_t ib,
                                  float ax_local, float ay_local, float az_local,
                                  uint8_t type) {
    // Rotate joint axis to parent world frame
    float ax_w, ay_w, az_w;
    qrot_to_world(bv, ia, ax_local, ay_local, az_local, ax_w, ay_w, az_w);

    if (type == static_cast<uint8_t>(JointType::Prismatic)) {
        // Current offset: projection of (pos_b - pos_a) onto axis
        return (bv.pos_x[ib]-bv.pos_x[ia])*ax_w
             + (bv.pos_y[ib]-bv.pos_y[ia])*ay_w
             + (bv.pos_z[ib]-bv.pos_z[ia])*az_w;
    }
    // Revolute: measure angle between reference directions
    // Pick a stable perpendicular to axis
    float px, py, pz;
    if (sycl::fabs(ax_local) < sycl::fabs(ay_local) &&
        sycl::fabs(ax_local) < sycl::fabs(az_local)) {
        // axis most orthogonal to X, so use X cross axis
        float len = sycl::sqrt(ay_local*ay_local + az_local*az_local);
        px = 0.f; py = -az_local/len; pz = ay_local/len;
    } else if (sycl::fabs(ay_local) < sycl::fabs(az_local)) {
        float len = sycl::sqrt(ax_local*ax_local + az_local*az_local);
        px = az_local/len; py = 0.f; pz = -ax_local/len;
    } else {
        float len = sycl::sqrt(ax_local*ax_local + ay_local*ay_local);
        px = -ay_local/len; py = ax_local/len; pz = 0.f;
    }
    // Ref direction in parent world frame and current direction in child world frame
    float rx, ry, rz; qrot_to_world(bv, ia, px, py, pz, rx, ry, rz);
    float cx, cy, cz; qrot_to_world(bv, ib, px, py, pz, cx, cy, cz);
    // Angle = atan2(cross · axis, dot)
    float cross_x = ry*cz - rz*cy, cross_y = rz*cx - rx*cz, cross_z = rx*cy - ry*cx;
    float sin_theta = cross_x*ax_w + cross_y*ay_w + cross_z*az_w;
    float cos_theta = rx*cx + ry*cy + rz*cz;
    return sycl::atan2(sin_theta, cos_theta);
}

// Apply a one-sided limit: if value < lo, push back above lo; if value > hi, push below hi.
// Implemented as a soft angular (revolute) or positional-along-axis (prismatic) constraint.
inline void apply_limit(const BodyView& bv, uint32_t ia, uint32_t ib,
                         float ax_local, float ay_local, float az_local,
                         float limit_lo, float limit_hi, float h,
                         uint8_t type) {
    if (limit_lo > limit_hi) return;  // limits disabled
    float val = joint_current_value(bv, ia, ib, ax_local, ay_local, az_local, type);
    float viol = 0.f;
    if      (val < limit_lo) viol = val - limit_lo;
    else if (val > limit_hi) viol = val - limit_hi;
    if (sycl::fabs(viol) < 1e-7f) return;

    float ax_w, ay_w, az_w;
    qrot_to_world(bv, ia, ax_local, ay_local, az_local, ax_w, ay_w, az_w);

    if (type == static_cast<uint8_t>(JointType::Prismatic)) {
        // Positional correction along axis
        float nx = ax_w, ny = ay_w, nz = az_w;
        float wa = gen_inv_mass(bv, ia, 0,0,0, nx, ny, nz);
        float wb = gen_inv_mass(bv, ib, 0,0,0, nx, ny, nz);
        float w  = wa + wb;
        if (w < 1e-10f) return;
        float dl = -viol / w;
        apply_pos_impulse(bv, ia, +1.f, dl, nx, ny, nz, 0,0,0);
        apply_pos_impulse(bv, ib, -1.f, dl, nx, ny, nz, 0,0,0);
    } else {
        // Angular correction: push the relative rotation back inside [lo, hi]
        float cx = viol * ax_w, cy = viol * ay_w, cz = viol * az_w;
        solve_ang_constraint(bv, ia, ib, cx, cy, cz, 0.f);
    }
    (void)h;
}

// ── PD motor velocity correction ──────────────────────────────────────────────
// Applied once per solve call (not per Gauss-Seidel iteration) for stability.
inline void apply_motor(const BodyView& bv, uint32_t ia, uint32_t ib,
                         float ax_local, float ay_local, float az_local,
                         float target_pos, float target_vel,
                         float stiffness, float damping, float h,
                         uint8_t type) {
    if (stiffness < 1e-6f && damping < 1e-6f) return;

    float ax_w, ay_w, az_w;
    qrot_to_world(bv, ia, ax_local, ay_local, az_local, ax_w, ay_w, az_w);

    // Angular (revolute) or linear (prismatic) relative velocity along axis
    float vel_rel;
    if (type == static_cast<uint8_t>(JointType::Prismatic)) {
        float vx = bv.vel_x[ib] - bv.vel_x[ia];
        float vy = bv.vel_y[ib] - bv.vel_y[ia];
        float vz = bv.vel_z[ib] - bv.vel_z[ia];
        vel_rel = vx*ax_w + vy*ay_w + vz*az_w;
    } else {
        float ox = bv.ang_x[ib] - bv.ang_x[ia];
        float oy = bv.ang_y[ib] - bv.ang_y[ia];
        float oz = bv.ang_z[ib] - bv.ang_z[ia];
        vel_rel = ox*ax_w + oy*ay_w + oz*az_w;
    }

    // PD: desired acceleration × h² → positional correction
    float pos_error = 0.f;
    if (stiffness > 1e-6f) {
        float val = joint_current_value(bv, ia, ib, ax_local, ay_local, az_local, type);
        pos_error = target_pos - val;
    }
    float vel_error = target_vel - vel_rel;

    // Effective angular / linear inverse mass along axis
    float ix_a, iy_a, iz_a, ix_b, iy_b, iz_b;
    if (type == static_cast<uint8_t>(JointType::Prismatic)) {
        world_inv_inertia(bv, ia, ax_w, ay_w, az_w, ix_a, iy_a, iz_a);
        world_inv_inertia(bv, ib, ax_w, ay_w, az_w, ix_b, iy_b, iz_b);
        float wa = bv.inv_mass[ia];
        float wb = bv.inv_mass[ib];
        float w  = wa + wb;
        if (w < 1e-10f) return;
        float j = (stiffness * pos_error * h + damping * vel_error) / w;
        bv.vel_x[ia] -= wa * j * ax_w; bv.vel_y[ia] -= wa * j * ay_w; bv.vel_z[ia] -= wa * j * az_w;
        bv.vel_x[ib] += wb * j * ax_w; bv.vel_y[ib] += wb * j * ay_w; bv.vel_z[ib] += wb * j * az_w;
    } else {
        world_inv_inertia(bv, ia, ax_w, ay_w, az_w, ix_a, iy_a, iz_a);
        world_inv_inertia(bv, ib, ax_w, ay_w, az_w, ix_b, iy_b, iz_b);
        float wa = ax_w*ix_a + ay_w*iy_a + az_w*iz_a;
        float wb = ax_w*ix_b + ay_w*iy_b + az_w*iz_b;
        float w  = wa + wb;
        if (w < 1e-10f) return;
        float j = (stiffness * pos_error * h + damping * vel_error) / w;
        bv.ang_x[ia] -= j * ix_a; bv.ang_y[ia] -= j * iy_a; bv.ang_z[ia] -= j * iz_a;
        bv.ang_x[ib] += j * ix_b; bv.ang_y[ib] += j * iy_b; bv.ang_z[ib] += j * iz_b;
    }
}

} // anonymous namespace

// ── XpbdSolver ───────────────────────────────────────────────────────────────

XpbdSolver::XpbdSolver(Stream& s, int n_iters, uint32_t max_contacts, float friction)
    : n_iters_(n_iters), friction_(friction), lambda_c_(s, max_contacts) {}

void XpbdSolver::solve(Stream& s, const ContactView& cv,
                       const JointView& jv, BodyView bv, float dt) {
    s.queue().memset(lambda_c_.data(), 0, lambda_c_.size() * sizeof(float));

    int    ni  = n_iters_;
    float  mu  = friction_;
    float* lc  = lambda_c_.data();

    parallel_for(s, 1, [cv, jv, bv, ni, mu, lc, dt](size_t) {
        uint32_t nc = *cv.n;
        uint32_t nj = jv.n;

        for (int iter = 0; iter < ni; ++iter) {

            // ── Contact constraints ──────────────────────────────────────────
            for (uint32_t c = 0; c < nc; ++c) {
                uint32_t ia = cv.body_a[c], ib = cv.body_b[c];
                float depth = cv.depth[c];
                if (depth <= 0.f) continue;

                float remaining = depth - lc[c];
                if (remaining <= 1e-6f) continue;

                float nx = cv.norm_x[c], ny = cv.norm_y[c], nz = cv.norm_z[c];
                float px = cv.pos_x[c],  py = cv.pos_y[c],  pz = cv.pos_z[c];

                float wa = bv.inv_mass[ia], wb = bv.inv_mass[ib];
                float rax = px - bv.pos_x[ia], ray = py - bv.pos_y[ia], raz = pz - bv.pos_z[ia];
                float rbx = px - bv.pos_x[ib], rby = py - bv.pos_y[ib], rbz = pz - bv.pos_z[ib];

                float raxnx = ray*nz - raz*ny, raxny = raz*nx - rax*nz, raxnz = rax*ny - ray*nx;
                float rbxnx = rby*nz - rbz*ny, rbxny = rbz*nx - rbx*nz, rbxnz = rbx*ny - rby*nx;

                float iIa_x, iIa_y, iIa_z, iIb_x, iIb_y, iIb_z;
                world_inv_inertia(bv, ia, raxnx, raxny, raxnz, iIa_x, iIa_y, iIa_z);
                world_inv_inertia(bv, ib, rbxnx, rbxny, rbxnz, iIb_x, iIb_y, iIb_z);

                float wa_ang = raxnx*iIa_x + raxny*iIa_y + raxnz*iIa_z;
                float wb_ang = rbxnx*iIb_x + rbxny*iIb_y + rbxnz*iIb_z;
                float w_total = wa + wb + wa_ang + wb_ang;
                if (w_total < 1e-10f) continue;

                float dl = remaining / w_total;
                lc[c] += remaining;

                // ── Normal position correction ───────────────────────────────
                bv.pos_x[ia] += wa * dl * nx; bv.pos_y[ia] += wa * dl * ny; bv.pos_z[ia] += wa * dl * nz;
                bv.pos_x[ib] -= wb * dl * nx; bv.pos_y[ib] -= wb * dl * ny; bv.pos_z[ib] -= wb * dl * nz;
                apply_ang_delta(bv, ia,  dl * iIa_x,  dl * iIa_y,  dl * iIa_z);
                apply_ang_delta(bv, ib, -dl * iIb_x, -dl * iIb_y, -dl * iIb_z);

                // ── Velocity correction ──────────────────────────────────────
                float oax=bv.ang_x[ia], oay=bv.ang_y[ia], oaz=bv.ang_z[ia];
                float obx=bv.ang_x[ib], oby=bv.ang_y[ib], obz=bv.ang_z[ib];
                float vcax = bv.vel_x[ia] + oay*raz - oaz*ray;
                float vcay = bv.vel_y[ia] + oaz*rax - oax*raz;
                float vcaz = bv.vel_z[ia] + oax*ray - oay*rax;
                float vcbx = bv.vel_x[ib] + oby*rbz - obz*rby;
                float vcby = bv.vel_y[ib] + obz*rbx - obx*rbz;
                float vcbz = bv.vel_z[ib] + obx*rby - oby*rbx;

                float vrel_x = vcax - vcbx, vrel_y = vcay - vcby, vrel_z = vcaz - vcbz;
                float vrel_n = vrel_x*nx + vrel_y*ny + vrel_z*nz;

                if (vrel_n < 0.f) {
                    float j_n = -vrel_n / w_total;
                    bv.vel_x[ia] += wa*j_n*nx; bv.vel_y[ia] += wa*j_n*ny; bv.vel_z[ia] += wa*j_n*nz;
                    bv.vel_x[ib] -= wb*j_n*nx; bv.vel_y[ib] -= wb*j_n*ny; bv.vel_z[ib] -= wb*j_n*nz;
                    float doa_x, doa_y, doa_z, dob_x, dob_y, dob_z;
                    world_inv_inertia(bv, ia, raxnx, raxny, raxnz, doa_x, doa_y, doa_z);
                    world_inv_inertia(bv, ib, rbxnx, rbxny, rbxnz, dob_x, dob_y, dob_z);
                    bv.ang_x[ia] += j_n*doa_x; bv.ang_y[ia] += j_n*doa_y; bv.ang_z[ia] += j_n*doa_z;
                    bv.ang_x[ib] -= j_n*dob_x; bv.ang_y[ib] -= j_n*dob_y; bv.ang_z[ib] -= j_n*dob_z;

                    // ── Coulomb friction ─────────────────────────────────────
                    if (mu > 0.f) {
                        float vt_x = vrel_x - vrel_n*nx;
                        float vt_y = vrel_y - vrel_n*ny;
                        float vt_z = vrel_z - vrel_n*nz;
                        float vt2  = vt_x*vt_x + vt_y*vt_y + vt_z*vt_z;
                        if (vt2 > 1e-12f) {
                            float vt_len = sycl::sqrt(vt2);
                            float tx = vt_x/vt_len, ty = vt_y/vt_len, tz = vt_z/vt_len;
                            float ratx = ray*tz - raz*ty, raty = raz*tx - rax*tz, ratz = rax*ty - ray*tx;
                            float rbtx = rby*tz - rbz*ty, rbty = rbz*tx - rbx*tz, rbtz = rbx*ty - rby*tx;
                            float fIa_x, fIa_y, fIa_z, fIb_x, fIb_y, fIb_z;
                            world_inv_inertia(bv, ia, ratx, raty, ratz, fIa_x, fIa_y, fIa_z);
                            world_inv_inertia(bv, ib, rbtx, rbty, rbtz, fIb_x, fIb_y, fIb_z);
                            float wt = wa + wb
                                     + ratx*fIa_x + raty*fIa_y + ratz*fIa_z
                                     + rbtx*fIb_x + rbty*fIb_y + rbtz*fIb_z;
                            if (wt > 1e-10f) {
                                float j_t = vt_len / wt;
                                j_t = sycl::fmin(j_t, mu * j_n);  // Coulomb clamp
                                bv.vel_x[ia] -= wa*j_t*tx; bv.vel_y[ia] -= wa*j_t*ty; bv.vel_z[ia] -= wa*j_t*tz;
                                bv.vel_x[ib] += wb*j_t*tx; bv.vel_y[ib] += wb*j_t*ty; bv.vel_z[ib] += wb*j_t*tz;
                                bv.ang_x[ia] -= j_t*fIa_x; bv.ang_y[ia] -= j_t*fIa_y; bv.ang_z[ia] -= j_t*fIa_z;
                                bv.ang_x[ib] += j_t*fIb_x; bv.ang_y[ib] += j_t*fIb_y; bv.ang_z[ib] += j_t*fIb_z;
                            }
                        }
                    }
                }
            } // end contact loop

            // ── Joint constraints ────────────────────────────────────────────
            for (uint32_t j = 0; j < nj; ++j) {
                uint32_t ia = jv.body_parent[j], ib = jv.body_child[j];
                uint8_t  jtype = jv.type[j];

                // World-frame lever arms (anchor positions in body frame → world frame)
                float rax, ray, raz;
                qrot_to_world(bv, ia,
                              jv.anchor_px[j], jv.anchor_py[j], jv.anchor_pz[j],
                              rax, ray, raz);
                float rbx, rby, rbz;
                qrot_to_world(bv, ib,
                              jv.anchor_cx[j], jv.anchor_cy[j], jv.anchor_cz[j],
                              rbx, rby, rbz);

                float alpha_pos = jv.compliance_pos[j] / (dt * dt);
                float alpha_ang = jv.compliance_ang[j] / (dt * dt);
                float alx = jv.axis_px[j], aly = jv.axis_py[j], alz = jv.axis_pz[j];

                if (jtype == static_cast<uint8_t>(JointType::Prismatic)) {
                    // Prismatic: fully constrained angular, 2-transverse positional
                    float axw, ayw, azw;
                    qrot_to_world(bv, ia, alx, aly, alz, axw, ayw, azw);
                    solve_fixed_ang(bv, ia, ib, alpha_ang);
                    solve_transverse_pos(bv, ia, ib, rax, ray, raz, rbx, rby, rbz,
                                         axw, ayw, azw, alpha_pos);
                    apply_limit(bv, ia, ib, alx, aly, alz,
                                jv.limit_lo[j], jv.limit_hi[j], dt, jtype);
                } else {
                    // Ball, Fixed, Revolute: all have 3D positional constraint
                    solve_pos_constraint(bv, ia, ib, rax, ray, raz, rbx, rby, rbz, alpha_pos);

                    if (jtype == static_cast<uint8_t>(JointType::Fixed)) {
                        solve_fixed_ang(bv, ia, ib, alpha_ang);
                    } else if (jtype == static_cast<uint8_t>(JointType::Revolute)) {
                        solve_revolute_ang(bv, ia, ib, alx, aly, alz, alpha_ang);
                        apply_limit(bv, ia, ib, alx, aly, alz,
                                    jv.limit_lo[j], jv.limit_hi[j], dt, jtype);
                    }
                    // Ball: no angular constraint
                }

                // PD motor (velocity correction, applied each iteration for implicit damping)
                if (jtype == static_cast<uint8_t>(JointType::Revolute) ||
                    jtype == static_cast<uint8_t>(JointType::Prismatic)) {
                    apply_motor(bv, ia, ib, alx, aly, alz,
                                jv.target_pos[j], jv.target_vel[j],
                                jv.stiffness[j], jv.damping[j], dt, jtype);
                }
            } // end joint loop

            // Normalize quaternions every iteration to prevent drift-induced NaN.
            for (uint32_t i = 0; i < bv.n; ++i) {
                float qw=bv.rot_w[i], qx=bv.rot_x[i], qy=bv.rot_y[i], qz=bv.rot_z[i];
                float inv_len = sycl::rsqrt(qw*qw + qx*qx + qy*qy + qz*qz);
                bv.rot_w[i] = qw*inv_len; bv.rot_x[i] = qx*inv_len;
                bv.rot_y[i] = qy*inv_len; bv.rot_z[i] = qz*inv_len;
            }

        } // end Gauss-Seidel iterations

    });
}

} // namespace dyphur
