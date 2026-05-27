#include <core/shape_store.hpp>
#include <cassert>

namespace dyphur {

ShapeStore::ShapeStore(Stream& s, uint32_t capacity)
    : cap_(capacity)
    , d_type_(s, capacity)
    , d_half_x_(s, capacity), d_half_y_(s, capacity), d_half_z_(s, capacity)
    , d_ext_id_(s, capacity)
{
    h_type_.reserve(capacity);
    h_half_x_.reserve(capacity);
    h_half_y_.reserve(capacity);
    h_half_z_.reserve(capacity);
    h_ext_id_.reserve(capacity);
}

uint32_t ShapeStore::add(const ShapeParams& p) {
    assert(count_ < cap_ && "ShapeStore capacity exceeded");
    h_type_.push_back(static_cast<uint32_t>(p.type));
    h_half_x_.push_back(p.half_x);
    h_half_y_.push_back(p.half_y);
    h_half_z_.push_back(p.half_z);
    h_ext_id_.push_back(p.ext_id);
    return count_++;
}

void ShapeStore::upload() {
    if (count_ == 0) return;
    d_type_.upload(h_type_.data(), count_);
    d_half_x_.upload(h_half_x_.data(), count_);
    d_half_y_.upload(h_half_y_.data(), count_);
    d_half_z_.upload(h_half_z_.data(), count_);
    d_ext_id_.upload(h_ext_id_.data(), count_);
}

ShapeView ShapeStore::view() noexcept {
    return { d_type_.data(), d_half_x_.data(), d_half_y_.data(), d_half_z_.data(),
             d_ext_id_.data(), count_ };
}

} // namespace dyphur
