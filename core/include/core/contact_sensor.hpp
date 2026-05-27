#pragma once
#include "narrowphase.hpp"
#include <compute/stream.hpp>
#include <cstdint>
#include <vector>

namespace dyphur {

// Per-shape contact event subscription.
//
// After each narrowphase run, call query() to download and filter all contacts
// that involve a body whose shape index matches shape_idx.  The filtered data
// is stored in host-side vectors and accessible as raw pointers for zero-copy
// numpy views from Python.
//
// Contacts are ordered by the original narrowphase output order; ties broken
// arbitrarily.  Query is blocking (calls s.wait() internally if the count has
// not already been fetched for this frame).
class ContactSensor {
public:
    explicit ContactSensor(uint32_t shape_idx) : shape_idx_(shape_idx) {}

    // Download all contacts from np, filter to those involving shape_idx_, and
    // store results in host buffers.  body_shapes[body_i] gives the shape index
    // of body i (from BodyStore::body_shapes()).
    // Returns the number of matching contacts stored.
    uint32_t query(Stream& s,
                   const Narrowphase& np,
                   const std::vector<uint32_t>& body_shapes);

    // Number of contacts found by the most recent query().
    uint32_t count() const noexcept { return count_; }

    uint32_t         shape_idx() const noexcept { return shape_idx_; }

    // Host-side contact arrays (valid after query(), size == count()).
    const uint32_t* body_a() const noexcept { return h_body_a_.data(); }
    const uint32_t* body_b() const noexcept { return h_body_b_.data(); }
    const float*    pos_x()  const noexcept { return h_pos_x_.data();  }
    const float*    pos_y()  const noexcept { return h_pos_y_.data();  }
    const float*    pos_z()  const noexcept { return h_pos_z_.data();  }
    const float*    norm_x() const noexcept { return h_norm_x_.data(); }
    const float*    norm_y() const noexcept { return h_norm_y_.data(); }
    const float*    norm_z() const noexcept { return h_norm_z_.data(); }
    const float*    depth()  const noexcept { return h_depth_.data();  }

private:
    uint32_t shape_idx_;
    uint32_t count_ = 0;

    std::vector<uint32_t> h_body_a_, h_body_b_;
    std::vector<float>    h_pos_x_,  h_pos_y_,  h_pos_z_;
    std::vector<float>    h_norm_x_, h_norm_y_, h_norm_z_;
    std::vector<float>    h_depth_;
};

} // namespace dyphur
