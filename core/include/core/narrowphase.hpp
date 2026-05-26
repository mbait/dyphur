#pragma once
#include "broadphase.hpp"
#include "body.hpp"
#include "shapes.hpp"
#include "contact_store.hpp"
#include <compute/stream.hpp>

namespace dyphur {

// Dispatches candidate pairs from the broadphase to primitive-vs-primitive
// contact tests and writes results to the internal ContactStore.
//
// Supported shape pairs (any ordering):
//   Sphere–Sphere : exact distance test
//   Sphere–Box    : closest-point query on OBB
//   Box–Box       : SAT (15 axes) + vertex-face manifold (up to 4 contacts)
//
// All submissions are async on s; call s.wait() before reading contacts.
class Narrowphase {
public:
    Narrowphase() = default;
    explicit Narrowphase(Stream& s, uint32_t max_contacts);

    // n_pairs must be downloaded from broadphase before calling.
    void run(Stream& s,
             const ContactPair* d_pairs, uint32_t n_pairs,
             const BodyView& bodies, const ShapeView& shapes);

    ContactView contacts()   noexcept { return store_.view(); }
    uint32_t download_count(Stream& s) const { return store_.download_count(s); }

private:
    ContactStore store_;
};

} // namespace dyphur
