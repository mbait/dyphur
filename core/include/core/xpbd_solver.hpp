#pragma once
#include "solver.hpp"

namespace dyphur {

// Sequential Gauss-Seidel XPBD solver.
//
// Processes contacts in a fixed order (deterministic) from a single GPU
// work-item. Runs n_iters position+velocity correction passes per call.
// Angular response is included (full rigid body, diagonal body-frame inertia).
// Restitution is zero (inelastic); friction is deferred to Phase 2.
class XpbdSolver final : public IConstraintSolver {
public:
    explicit XpbdSolver(int n_iters = 10) : n_iters_(n_iters) {}

    void solve(Stream& s, const ContactView& contacts,
               BodyView bodies, float dt) override;

private:
    int n_iters_;
};

} // namespace dyphur
