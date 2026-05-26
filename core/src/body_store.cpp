#include <core/body_store.hpp>

namespace dyphur {

BodyStore::BodyStore(Stream& s, uint32_t capacity)
    : cap_(capacity)
    , d_pos_x_(s, capacity), d_pos_y_(s, capacity), d_pos_z_(s, capacity)
    , d_rot_w_(s, capacity), d_rot_x_(s, capacity)
    , d_rot_y_(s, capacity), d_rot_z_(s, capacity)
    , d_vel_x_(s, capacity), d_vel_y_(s, capacity), d_vel_z_(s, capacity)
    , d_ang_x_(s, capacity), d_ang_y_(s, capacity), d_ang_z_(s, capacity)
    , d_inv_mass_(s, capacity)
    , d_iI_xx_(s, capacity), d_iI_yy_(s, capacity), d_iI_zz_(s, capacity)
    , d_iI_xy_(s, capacity), d_iI_xz_(s, capacity), d_iI_yz_(s, capacity)
    , d_shape_(s, capacity), d_flags_(s, capacity)
{
    h_pos_x_.reserve(capacity);  h_pos_y_.reserve(capacity);  h_pos_z_.reserve(capacity);
    h_rot_w_.reserve(capacity);  h_rot_x_.reserve(capacity);
    h_rot_y_.reserve(capacity);  h_rot_z_.reserve(capacity);
    h_vel_x_.reserve(capacity);  h_vel_y_.reserve(capacity);  h_vel_z_.reserve(capacity);
    h_ang_x_.reserve(capacity);  h_ang_y_.reserve(capacity);  h_ang_z_.reserve(capacity);
    h_inv_mass_.reserve(capacity);
    h_iI_xx_.reserve(capacity);  h_iI_yy_.reserve(capacity);  h_iI_zz_.reserve(capacity);
    h_iI_xy_.reserve(capacity);  h_iI_xz_.reserve(capacity);  h_iI_yz_.reserve(capacity);
    h_shape_.reserve(capacity);  h_flags_.reserve(capacity);
}

uint32_t BodyStore::add(const BodyParams& p) {
    assert(count_ < cap_ && "BodyStore capacity exceeded");

    h_pos_x_.push_back(p.position.x);
    h_pos_y_.push_back(p.position.y);
    h_pos_z_.push_back(p.position.z);

    h_rot_w_.push_back(p.rotation.w);
    h_rot_x_.push_back(p.rotation.xyz.x);
    h_rot_y_.push_back(p.rotation.xyz.y);
    h_rot_z_.push_back(p.rotation.xyz.z);

    h_vel_x_.push_back(p.linear_velocity.x);
    h_vel_y_.push_back(p.linear_velocity.y);
    h_vel_z_.push_back(p.linear_velocity.z);

    h_ang_x_.push_back(p.angular_velocity.x);
    h_ang_y_.push_back(p.angular_velocity.y);
    h_ang_z_.push_back(p.angular_velocity.z);

    bool is_static = (p.flags & BodyFlag::Static) || p.mass <= 0.f;
    float inv_m = is_static ? 0.f : 1.f / p.mass;
    h_inv_mass_.push_back(inv_m);

    if (is_static) {
        h_iI_xx_.push_back(0.f); h_iI_yy_.push_back(0.f); h_iI_zz_.push_back(0.f);
        h_iI_xy_.push_back(0.f); h_iI_xz_.push_back(0.f); h_iI_yz_.push_back(0.f);
    } else {
        Mat3f inv_I = p.inertia.inverse();
        h_iI_xx_.push_back(inv_I(0, 0));
        h_iI_yy_.push_back(inv_I(1, 1));
        h_iI_zz_.push_back(inv_I(2, 2));
        h_iI_xy_.push_back(inv_I(0, 1));
        h_iI_xz_.push_back(inv_I(0, 2));
        h_iI_yz_.push_back(inv_I(1, 2));
    }

    h_shape_.push_back(p.shape_handle);
    h_flags_.push_back(is_static ? (p.flags | BodyFlag::Static) : p.flags);

    return count_++;
}

void BodyStore::upload() {
    if (count_ == 0) return;
    const uint32_t n = count_;
    d_pos_x_.upload(h_pos_x_.data(), n);
    d_pos_y_.upload(h_pos_y_.data(), n);
    d_pos_z_.upload(h_pos_z_.data(), n);
    d_rot_w_.upload(h_rot_w_.data(), n);
    d_rot_x_.upload(h_rot_x_.data(), n);
    d_rot_y_.upload(h_rot_y_.data(), n);
    d_rot_z_.upload(h_rot_z_.data(), n);
    d_vel_x_.upload(h_vel_x_.data(), n);
    d_vel_y_.upload(h_vel_y_.data(), n);
    d_vel_z_.upload(h_vel_z_.data(), n);
    d_ang_x_.upload(h_ang_x_.data(), n);
    d_ang_y_.upload(h_ang_y_.data(), n);
    d_ang_z_.upload(h_ang_z_.data(), n);
    d_inv_mass_.upload(h_inv_mass_.data(), n);
    d_iI_xx_.upload(h_iI_xx_.data(), n);
    d_iI_yy_.upload(h_iI_yy_.data(), n);
    d_iI_zz_.upload(h_iI_zz_.data(), n);
    d_iI_xy_.upload(h_iI_xy_.data(), n);
    d_iI_xz_.upload(h_iI_xz_.data(), n);
    d_iI_yz_.upload(h_iI_yz_.data(), n);
    d_shape_.upload(h_shape_.data(), n);
    d_flags_.upload(h_flags_.data(), n);
}

BodyView BodyStore::view() noexcept {
    return {
        d_pos_x_.data(), d_pos_y_.data(), d_pos_z_.data(),
        d_rot_w_.data(), d_rot_x_.data(), d_rot_y_.data(), d_rot_z_.data(),
        d_vel_x_.data(), d_vel_y_.data(), d_vel_z_.data(),
        d_ang_x_.data(), d_ang_y_.data(), d_ang_z_.data(),
        d_inv_mass_.data(),
        d_iI_xx_.data(), d_iI_yy_.data(), d_iI_zz_.data(),
        d_iI_xy_.data(), d_iI_xz_.data(), d_iI_yz_.data(),
        d_shape_.data(), d_flags_.data(),
        count_
    };
}

} // namespace dyphur
