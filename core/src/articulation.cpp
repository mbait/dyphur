#include <core/joint_store.hpp>

namespace dyphur {

JointStore::JointStore(Stream& s, uint32_t capacity)
    : cap_(capacity)
    , d_body_parent_(s, capacity), d_body_child_(s, capacity)
    , d_anchor_px_(s, capacity), d_anchor_py_(s, capacity), d_anchor_pz_(s, capacity)
    , d_anchor_cx_(s, capacity), d_anchor_cy_(s, capacity), d_anchor_cz_(s, capacity)
    , d_axis_px_(s, capacity),   d_axis_py_(s, capacity),   d_axis_pz_(s, capacity)
    , d_limit_lo_(s, capacity),  d_limit_hi_(s, capacity)
    , d_target_pos_(s, capacity), d_target_vel_(s, capacity)
    , d_stiffness_(s, capacity),  d_damping_(s, capacity)
    , d_compliance_pos_(s, capacity), d_compliance_ang_(s, capacity)
    , d_type_(s, capacity)
{
    h_body_parent_.reserve(capacity);  h_body_child_.reserve(capacity);
    h_anchor_px_.reserve(capacity);    h_anchor_py_.reserve(capacity);    h_anchor_pz_.reserve(capacity);
    h_anchor_cx_.reserve(capacity);    h_anchor_cy_.reserve(capacity);    h_anchor_cz_.reserve(capacity);
    h_axis_px_.reserve(capacity);      h_axis_py_.reserve(capacity);      h_axis_pz_.reserve(capacity);
    h_limit_lo_.reserve(capacity);     h_limit_hi_.reserve(capacity);
    h_target_pos_.reserve(capacity);   h_target_vel_.reserve(capacity);
    h_stiffness_.reserve(capacity);    h_damping_.reserve(capacity);
    h_compliance_pos_.reserve(capacity); h_compliance_ang_.reserve(capacity);
    h_type_.reserve(capacity);
}

uint32_t JointStore::add(const JointParams& p) {
    assert(count_ < cap_ && "JointStore capacity exceeded");

    h_body_parent_.push_back(p.body_parent);
    h_body_child_.push_back(p.body_child);

    h_anchor_px_.push_back(p.anchor_parent.x);
    h_anchor_py_.push_back(p.anchor_parent.y);
    h_anchor_pz_.push_back(p.anchor_parent.z);

    h_anchor_cx_.push_back(p.anchor_child.x);
    h_anchor_cy_.push_back(p.anchor_child.y);
    h_anchor_cz_.push_back(p.anchor_child.z);

    h_axis_px_.push_back(p.axis.x);
    h_axis_py_.push_back(p.axis.y);
    h_axis_pz_.push_back(p.axis.z);

    h_limit_lo_.push_back(p.limit_lo);
    h_limit_hi_.push_back(p.limit_hi);

    h_target_pos_.push_back(p.target_pos);
    h_target_vel_.push_back(p.target_vel);
    h_stiffness_.push_back(p.stiffness);
    h_damping_.push_back(p.damping);

    h_compliance_pos_.push_back(p.compliance_pos);
    h_compliance_ang_.push_back(p.compliance_ang);

    h_type_.push_back(static_cast<uint8_t>(p.type));

    return count_++;
}

void JointStore::set_targets(const float* pos, const float* vel) {
    std::copy(pos, pos + count_, h_target_pos_.begin());
    std::copy(vel, vel + count_, h_target_vel_.begin());
    d_target_pos_.upload(h_target_pos_.data(), count_);
    d_target_vel_.upload(h_target_vel_.data(), count_);
}

void JointStore::upload() {
    if (count_ == 0) return;
    const uint32_t n = count_;
    d_body_parent_.upload(h_body_parent_.data(), n);
    d_body_child_.upload(h_body_child_.data(), n);
    d_anchor_px_.upload(h_anchor_px_.data(), n);
    d_anchor_py_.upload(h_anchor_py_.data(), n);
    d_anchor_pz_.upload(h_anchor_pz_.data(), n);
    d_anchor_cx_.upload(h_anchor_cx_.data(), n);
    d_anchor_cy_.upload(h_anchor_cy_.data(), n);
    d_anchor_cz_.upload(h_anchor_cz_.data(), n);
    d_axis_px_.upload(h_axis_px_.data(), n);
    d_axis_py_.upload(h_axis_py_.data(), n);
    d_axis_pz_.upload(h_axis_pz_.data(), n);
    d_limit_lo_.upload(h_limit_lo_.data(), n);
    d_limit_hi_.upload(h_limit_hi_.data(), n);
    d_target_pos_.upload(h_target_pos_.data(), n);
    d_target_vel_.upload(h_target_vel_.data(), n);
    d_stiffness_.upload(h_stiffness_.data(), n);
    d_damping_.upload(h_damping_.data(), n);
    d_compliance_pos_.upload(h_compliance_pos_.data(), n);
    d_compliance_ang_.upload(h_compliance_ang_.data(), n);
    d_type_.upload(h_type_.data(), n);
}

JointView JointStore::view() noexcept {
    return {
        d_body_parent_.data(), d_body_child_.data(),
        d_anchor_px_.data(), d_anchor_py_.data(), d_anchor_pz_.data(),
        d_anchor_cx_.data(), d_anchor_cy_.data(), d_anchor_cz_.data(),
        d_axis_px_.data(),   d_axis_py_.data(),   d_axis_pz_.data(),
        d_limit_lo_.data(),  d_limit_hi_.data(),
        d_target_pos_.data(), d_target_vel_.data(),
        d_stiffness_.data(),  d_damping_.data(),
        d_compliance_pos_.data(), d_compliance_ang_.data(),
        d_type_.data(),
        count_
    };
}

} // namespace dyphur
