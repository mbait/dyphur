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
//
// Parallelism: small scenes (≤ kSerialThreshold constraints) use a single
// work-item sequential Gauss-Seidel pass (the original path — preserves its
// results/goldens). Large scenes use a graph-colored parallel pass: constraints
// are partitioned so no two in a colour share a dynamic body, each colour solves
// in parallel (disjoint writes ⇒ no atomics), and colours run in queue order
// (Gauss-Seidel across colours). Deterministic by construction on either path.
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

    // Constraint count at/below which the sequential path is used.
    static constexpr uint32_t kSerialThreshold = 256;

    // Force the single-work-item sequential path regardless of scene size. Used as
    // a determinism oracle / cross-check for the colored path in tests; not for
    // production (it does not scale).
    void set_force_serial(bool v) { force_serial_ = v; }

private:
    // Ensure the colouring scratch buffers fit n_constraints / n_bodies.
    void ensure_capacity(Stream& s, uint32_t n_constraints, uint32_t n_bodies);

    int              n_iters_;
    float            friction_;
    bool             force_serial_ = false;
    Buffer<float>    lambda_c_;
    // Graph-colouring scratch (allocated lazily / grown on demand).
    Buffer<uint32_t> color_;      // [n_constraints] colour per constraint
    Buffer<uint64_t> body_mask_;  // [n_bodies] used-colour bitmask (≤64 colours)
    Buffer<uint32_t> meta_;       // [2] = {n_colors, overflow_flag}
};

} // namespace dyphur
