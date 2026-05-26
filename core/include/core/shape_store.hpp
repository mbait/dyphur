#pragma once
#include "shapes.hpp"
#include <compute/buffer.hpp>
#include <compute/stream.hpp>
#include <vector>

namespace dyphur {

class ShapeStore {
public:
    ShapeStore() = default;
    ShapeStore(Stream& s, uint32_t capacity);

    uint32_t  add(const ShapeParams& p);
    void      upload();
    ShapeView view() noexcept;
    uint32_t  count()    const noexcept { return count_; }
    uint32_t  capacity() const noexcept { return cap_; }

private:
    Buffer<uint32_t> d_type_;
    Buffer<float>    d_half_x_, d_half_y_, d_half_z_;
    std::vector<uint32_t> h_type_;
    std::vector<float>    h_half_x_, h_half_y_, h_half_z_;
    uint32_t count_ = 0, cap_ = 0;
};

} // namespace dyphur
