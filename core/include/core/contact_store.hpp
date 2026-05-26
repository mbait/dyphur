#pragma once
#include "contact.hpp"
#include <compute/buffer.hpp>
#include <compute/stream.hpp>

namespace dyphur {

class ContactStore {
public:
    ContactStore() = default;
    ContactStore(Stream& s, uint32_t capacity);

    void        reset(Stream& s);  // zero device count (async)
    ContactView view() noexcept;
    uint32_t    capacity()       const noexcept { return cap_; }
    uint32_t    download_count(Stream& s) const;

private:
    Buffer<uint32_t> d_body_a_, d_body_b_;
    Buffer<float>    d_pos_x_,  d_pos_y_,  d_pos_z_;
    Buffer<float>    d_norm_x_, d_norm_y_, d_norm_z_;
    Buffer<float>    d_depth_;
    Buffer<uint32_t> d_count_;
    uint32_t cap_ = 0;
};

} // namespace dyphur
