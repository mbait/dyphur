#include <core/integrator.hpp>
#include <compute/kernel.hpp>
#include <sycl/sycl.hpp>

namespace dyphur {

void integrate(Stream& s, BodyView bodies, const IntegratorParams& params) {
    if (bodies.n == 0) return;

    const float sub_dt   = params.dt / float(params.n_substeps);
    const float gx       = params.gravity.x;
    const float gy       = params.gravity.y;
    const float gz       = params.gravity.z;
    const float half_dt  = 0.5f * sub_dt;
    const uint32_t n     = bodies.n;

    for (int sub = 0; sub < params.n_substeps; ++sub) {
        parallel_for(s, n, [=](size_t i) {
            // Skip static and kinematic bodies (inv_mass == 0).
            const float inv_m = bodies.inv_mass[i];
            if (inv_m == 0.f) return;
            // Skip sleeping islands.
            if (bodies.flags[i] & BodyFlag::Sleeping) return;

            // ── Linear ──────────────────────────────────────────────────────
            // Symplectic Euler: update velocity first, then position.
            float vx = bodies.vel_x[i] + gx * sub_dt;
            float vy = bodies.vel_y[i] + gy * sub_dt;
            float vz = bodies.vel_z[i] + gz * sub_dt;

            bodies.vel_x[i] = vx;
            bodies.vel_y[i] = vy;
            bodies.vel_z[i] = vz;

            bodies.pos_x[i] += vx * sub_dt;
            bodies.pos_y[i] += vy * sub_dt;
            bodies.pos_z[i] += vz * sub_dt;

            // ── Angular ─────────────────────────────────────────────────────
            // q_dot = 0.5 * Quat(0, ang) * q
            // q_new = normalize(q + sub_dt * q_dot)
            const float ox = bodies.ang_x[i];
            const float oy = bodies.ang_y[i];
            const float oz = bodies.ang_z[i];

            float qw = bodies.rot_w[i];
            float qx = bodies.rot_x[i];
            float qy = bodies.rot_y[i];
            float qz = bodies.rot_z[i];

            // Quat(0, omega) * q (pure-quat product)
            const float dpw = -(ox*qx + oy*qy + oz*qz);
            const float dpx =  (ox*qw + oy*qz - oz*qy);
            const float dpy =  (oy*qw + oz*qx - ox*qz);
            const float dpz =  (oz*qw + ox*qy - oy*qx);

            qw += half_dt * dpw;
            qx += half_dt * dpx;
            qy += half_dt * dpy;
            qz += half_dt * dpz;

            // Renormalize (guards against drift over many steps).
            const float inv_n = sycl::rsqrt(qw*qw + qx*qx + qy*qy + qz*qz);
            bodies.rot_w[i] = qw * inv_n;
            bodies.rot_x[i] = qx * inv_n;
            bodies.rot_y[i] = qy * inv_n;
            bodies.rot_z[i] = qz * inv_n;
        });
        // Future: solver->solve(s, bodies, sub_dt) goes here.
    }
}

} // namespace dyphur
