#include <core/xpbd_solver.hpp>
#include <compute/kernel.hpp>
#include <sycl/sycl.hpp>

namespace dyphur {
namespace {

// Apply world-space inverse inertia to vector v for body bi.
// I_world_inv * v = R * I_body_inv * R^T * v
inline void world_inv_inertia(const BodyView& bv, uint32_t bi,
                               float vx, float vy, float vz,
                               float& ox, float& oy, float& oz) {
    float qw = bv.rot_w[bi], qx = bv.rot_x[bi];
    float qy = bv.rot_y[bi], qz = bv.rot_z[bi];
    float R00=1.f-2.f*(qy*qy+qz*qz), R01=2.f*(qx*qy+qz*qw), R02=2.f*(qx*qz-qy*qw);
    float R10=2.f*(qx*qy-qz*qw),     R11=1.f-2.f*(qx*qx+qz*qz), R12=2.f*(qy*qz+qx*qw);
    float R20=2.f*(qx*qz+qy*qw),     R21=2.f*(qy*qz-qx*qw),     R22=1.f-2.f*(qx*qx+qy*qy);
    float lx = R00*vx + R10*vy + R20*vz;
    float ly = R01*vx + R11*vy + R21*vz;
    float lz = R02*vx + R12*vy + R22*vz;
    float ilx = bv.iI_xx[bi]*lx + bv.iI_xy[bi]*ly + bv.iI_xz[bi]*lz;
    float ily = bv.iI_xy[bi]*lx + bv.iI_yy[bi]*ly + bv.iI_yz[bi]*lz;
    float ilz = bv.iI_xz[bi]*lx + bv.iI_yz[bi]*ly + bv.iI_zz[bi]*lz;
    ox = R00*ilx + R01*ily + R02*ilz;
    oy = R10*ilx + R11*ily + R12*ilz;
    oz = R20*ilx + R21*ily + R22*ilz;
}

} // anonymous namespace

XpbdSolver::XpbdSolver(Stream& s, int n_iters, uint32_t max_contacts)
    : n_iters_(n_iters), lambda_c_(s, max_contacts) {}

void XpbdSolver::solve(Stream& s, const ContactView& cv,
                       BodyView bv, float /*dt*/) {
    // Zero accumulated-lambda buffer before each solve.
    s.queue().memset(lambda_c_.data(), 0, lambda_c_.size() * sizeof(float));

    int    ni = n_iters_;
    float* lc = lambda_c_.data();

    parallel_for(s, 1, [cv, bv, ni, lc](size_t) {
        uint32_t nc = *cv.n;

        for (int iter = 0; iter < ni; ++iter) {
            for (uint32_t c = 0; c < nc; ++c) {
                uint32_t ia = cv.body_a[c], ib = cv.body_b[c];
                float depth = cv.depth[c];
                if (depth <= 0.f) continue;

                // Accumulated constraint resolution: skip if already resolved.
                // lc[c] tracks how much "depth" has been resolved this solve call.
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

                float delta_lambda = remaining / w_total;
                // Mark this contact as fully resolved so later iterations skip it.
                lc[c] += remaining;

                // ── Position correction ──────────────────────────────────────
                bv.pos_x[ia] += wa * delta_lambda * nx;
                bv.pos_y[ia] += wa * delta_lambda * ny;
                bv.pos_z[ia] += wa * delta_lambda * nz;
                bv.pos_x[ib] -= wb * delta_lambda * nx;
                bv.pos_y[ib] -= wb * delta_lambda * ny;
                bv.pos_z[ib] -= wb * delta_lambda * nz;

                // ── Angular position correction: Δq = ½ [0, I^-1(r×(Δλn))] ⊗ q ──
                {
                    float ax =  delta_lambda * iIa_x, ay =  delta_lambda * iIa_y, az =  delta_lambda * iIa_z;
                    float qw = bv.rot_w[ia], qx = bv.rot_x[ia];
                    float qy = bv.rot_y[ia], qz = bv.rot_z[ia];
                    bv.rot_w[ia] += 0.5f*(-ax*qx - ay*qy - az*qz);
                    bv.rot_x[ia] += 0.5f*( ax*qw + ay*qz - az*qy);
                    bv.rot_y[ia] += 0.5f*(-ax*qz + ay*qw + az*qx);
                    bv.rot_z[ia] += 0.5f*( ax*qy - ay*qx + az*qw);
                }
                {
                    float ax = -delta_lambda * iIb_x, ay = -delta_lambda * iIb_y, az = -delta_lambda * iIb_z;
                    float qw = bv.rot_w[ib], qx = bv.rot_x[ib];
                    float qy = bv.rot_y[ib], qz = bv.rot_z[ib];
                    bv.rot_w[ib] += 0.5f*(-ax*qx - ay*qy - az*qz);
                    bv.rot_x[ib] += 0.5f*( ax*qw + ay*qz - az*qy);
                    bv.rot_y[ib] += 0.5f*(-ax*qz + ay*qw + az*qx);
                    bv.rot_z[ib] += 0.5f*( ax*qy - ay*qx + az*qw);
                }

                // ── Velocity correction (e=0 restitution) ───────────────────
                float oax=bv.ang_x[ia], oay=bv.ang_y[ia], oaz=bv.ang_z[ia];
                float obx=bv.ang_x[ib], oby=bv.ang_y[ib], obz=bv.ang_z[ib];
                float vcax = bv.vel_x[ia] + oay*raz - oaz*ray;
                float vcay = bv.vel_y[ia] + oaz*rax - oax*raz;
                float vcaz = bv.vel_z[ia] + oax*ray - oay*rax;
                float vcbx = bv.vel_x[ib] + oby*rbz - obz*rby;
                float vcby = bv.vel_y[ib] + obz*rbx - obx*rbz;
                float vcbz = bv.vel_z[ib] + obx*rby - oby*rbx;
                float vrel_n = (vcax-vcbx)*nx + (vcay-vcby)*ny + (vcaz-vcbz)*nz;

                if (vrel_n < 0.f) {
                    float j = -vrel_n / w_total;
                    bv.vel_x[ia] += wa*j*nx; bv.vel_y[ia] += wa*j*ny; bv.vel_z[ia] += wa*j*nz;
                    bv.vel_x[ib] -= wb*j*nx; bv.vel_y[ib] -= wb*j*ny; bv.vel_z[ib] -= wb*j*nz;
                    float doa_x, doa_y, doa_z, dob_x, dob_y, dob_z;
                    world_inv_inertia(bv, ia, raxnx, raxny, raxnz, doa_x, doa_y, doa_z);
                    world_inv_inertia(bv, ib, rbxnx, rbxny, rbxnz, dob_x, dob_y, dob_z);
                    bv.ang_x[ia] += j*doa_x; bv.ang_y[ia] += j*doa_y; bv.ang_z[ia] += j*doa_z;
                    bv.ang_x[ib] -= j*dob_x; bv.ang_y[ib] -= j*dob_y; bv.ang_z[ib] -= j*dob_z;
                }
            }
        }

        // Normalise all quaternions after corrections.
        for (uint32_t i = 0; i < bv.n; ++i) {
            float qw=bv.rot_w[i], qx=bv.rot_x[i], qy=bv.rot_y[i], qz=bv.rot_z[i];
            float inv_len = sycl::rsqrt(qw*qw + qx*qx + qy*qy + qz*qz);
            bv.rot_w[i] = qw*inv_len; bv.rot_x[i] = qx*inv_len;
            bv.rot_y[i] = qy*inv_len; bv.rot_z[i] = qz*inv_len;
        }
    });
}

} // namespace dyphur
