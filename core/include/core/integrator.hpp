#pragma once
#include "body.hpp"
#include <compute/stream.hpp>

namespace dyphur {

struct IntegratorParams {
    Vec3f gravity    = {0.f, -9.81f, 0.f};
    float dt         = 1.f / 60.f;  // full step duration (seconds)
    int   n_substeps = 4;           // substep count; each substep = dt / n_substeps
};

// Advance all dynamic bodies by params.dt using semi-implicit (symplectic) Euler.
//
// Per substep, for each body with inv_mass > 0 and not sleeping:
//   vel   += gravity * sub_dt
//   pos   += vel    * sub_dt        (position updated with already-updated velocity)
//   rot    = normalize(rot + 0.5 * sub_dt * Quat(0, ang) * rot)
//
// Static (inv_mass == 0) and sleeping bodies are untouched.
// Substeps are issued as separate kernel launches; constraint solving can be
// inserted between them when the solver is wired in.
//
// Submissions are async on stream s; caller calls s.wait() when needed.
void integrate(Stream& s, BodyView bodies, const IntegratorParams& params);

} // namespace dyphur
