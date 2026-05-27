#pragma once
#include "contact.hpp"
#include "articulation.hpp"
#include "body.hpp"
#include <compute/stream.hpp>

namespace dyphur {

// Solver-agnostic interface.
// Implementations must be safe to call every frame; they own no per-frame state.
class IConstraintSolver {
public:
    virtual ~IConstraintSolver() = default;

    // Resolve contacts and joints in-place, then update velocities.
    // Pass JointView{} (n == 0) if there are no joints this frame.
    virtual void solve(Stream& s,
                       const ContactView& contacts,
                       const JointView&   joints,
                       BodyView           bodies,
                       float              dt) = 0;
};

} // namespace dyphur
