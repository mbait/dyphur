#pragma once
#include "solver.hpp"
#include <compute/buffer.hpp>

namespace dyphur {

// Sequential Gauss-Seidel XPBD solver.
//
// Each solve call processes contacts and joints together in n_iters passes.
// Angular response is full rigid body (body-frame diagonal inertia).
// Restitution: zero (inelastic). Friction: Coulomb, global coefficient mu.
//
// Joint types handled:
//   Fixed    — positional (3 DOF) + quaternion angular (3 DOF)
//   Ball     — positional (3 DOF) only
//   Revolute — positional (3 DOF) + axis-alignment angular (2 DOF) + optional PD motor + limits
//   Prismatic— quaternion angular (3 DOF) + transverse positional (2 DOF) + optional PD motor + limits
//
// lambda_c_ accumulates per-contact resolved depth so that each contact is
// corrected at most once per solve call (prevents over-application in manifolds
// with multiple contact points per body pair).
class XpbdSolver final : public IConstraintSolver {
public:
    explicit XpbdSolver(Stream& s,
                        int      n_iters      = 10,
                        uint32_t max_contacts = 16384,
                        float    friction     = 0.5f);

    void solve(Stream& s,
               const ContactView& contacts,
               const JointView&   joints,
               BodyView           bodies,
               float              dt) override;

private:
    int           n_iters_;
    float         friction_;
    Buffer<float> lambda_c_;
};

} // namespace dyphur
