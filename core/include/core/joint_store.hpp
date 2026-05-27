#pragma once
#include "articulation.hpp"
#include <compute/buffer.hpp>
#include <compute/stream.hpp>
#include <vector>
#include <cassert>

namespace dyphur {

// Host-side owner of the device-resident joint SoA arrays.
//
// Lifecycle:
//   1. JointStore store(stream, capacity)
//   2. store.add(params) ...
//   3. store.upload()
//   4. stream.wait()
//   5. use store.view() in solvers
class JointStore {
public:
    JointStore() = default;
    JointStore(Stream& s, uint32_t capacity);

    JointStore(const JointStore&)            = delete;
    JointStore& operator=(const JointStore&) = delete;
    JointStore(JointStore&&)                 = default;
    JointStore& operator=(JointStore&&)      = default;

    uint32_t  add(const JointParams& p);
    void      upload();
    void      set_targets(const float* pos, const float* vel);  // update PD targets and re-upload
    JointView view() noexcept;

    uint32_t count()    const noexcept { return count_; }
    uint32_t capacity() const noexcept { return cap_; }

private:
    Buffer<uint32_t> d_body_parent_, d_body_child_;
    Buffer<float>    d_anchor_px_, d_anchor_py_, d_anchor_pz_;
    Buffer<float>    d_anchor_cx_, d_anchor_cy_, d_anchor_cz_;
    Buffer<float>    d_axis_px_,   d_axis_py_,   d_axis_pz_;
    Buffer<float>    d_limit_lo_,  d_limit_hi_;
    Buffer<float>    d_target_pos_, d_target_vel_;
    Buffer<float>    d_stiffness_,  d_damping_;
    Buffer<float>    d_compliance_pos_, d_compliance_ang_;
    Buffer<uint8_t>  d_type_;

    std::vector<uint32_t> h_body_parent_, h_body_child_;
    std::vector<float>    h_anchor_px_, h_anchor_py_, h_anchor_pz_;
    std::vector<float>    h_anchor_cx_, h_anchor_cy_, h_anchor_cz_;
    std::vector<float>    h_axis_px_,   h_axis_py_,   h_axis_pz_;
    std::vector<float>    h_limit_lo_,  h_limit_hi_;
    std::vector<float>    h_target_pos_, h_target_vel_;
    std::vector<float>    h_stiffness_,  h_damping_;
    std::vector<float>    h_compliance_pos_, h_compliance_ang_;
    std::vector<uint8_t>  h_type_;

    uint32_t count_ = 0;
    uint32_t cap_   = 0;
};

} // namespace dyphur
