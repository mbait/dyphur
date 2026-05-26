#pragma once
#include "solver.hpp"
#include <compute/buffer.hpp>

namespace dyphur {

// Sequential Gauss-Seidel XPBD solver.
//
// Processes contacts in a fixed order (deterministic) from a single GPU
// work-item. Runs n_iters position+velocity correction passes per call.
// Angular response is included (full rigid body, diagonal body-frame inertia).
// Restitution is zero (inelastic); friction is deferred to Phase 2.
//
// lambda_c_ accumulates per-contact resolved depth across iterations so that
// each contact is corrected at most once per solve call (prevents the 10×
// over-application that would otherwise occur with multi-iteration GS and
// vertex-face manifolds producing multiple contacts per body pair).
class XpbdSolver final : public IConstraintSolver {
public:
    explicit XpbdSolver(Stream& s, int n_iters = 10,
                        uint32_t max_contacts = 16384);

    void solve(Stream& s, const ContactView& contacts,
               BodyView bodies, float dt) override;

private:
    int           n_iters_;
    Buffer<float> lambda_c_;
};

} // namespace dyphur
