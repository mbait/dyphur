#pragma once
#include "contact.hpp"
#include "body.hpp"
#include <compute/stream.hpp>

namespace dyphur {

// Solver-agnostic interface.
// Implementations must be safe to call every frame; they own no per-frame state.
class IConstraintSolver {
public:
    virtual ~IConstraintSolver() = default;

    // Resolve contacts in-place: modify bodies.pos and bodies.rot to satisfy
    // non-penetration, then update velocities for zero restitution.
    // contacts.n is a device pointer; the solver reads it on-device.
    virtual void solve(Stream& s, const ContactView& contacts,
                       BodyView bodies, float dt) = 0;
};

} // namespace dyphur
