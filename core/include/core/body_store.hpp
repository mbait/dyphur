#pragma once
#include "body.hpp"
#include <compute/buffer.hpp>
#include <compute/stream.hpp>
#include <vector>
#include <cassert>

namespace dyphur {

// Host-side owner of the device-resident body SoA arrays.
//
// Lifecycle:
//   1. BodyStore store(stream, capacity)   — allocates device buffers
//   2. store.add(params)  ...              — populate host shadow arrays
//   3. store.upload()                      — bulk DMA to device (async on stream)
//   4. stream.wait()                       — ensure device sees the data
//   5. use store.view() in kernels every step; modify via kernels in-place
//
// Not copyable (owns device memory). The Stream passed to the constructor must
// outlive this object.
class BodyStore {
public:
    BodyStore() = default;
    BodyStore(Stream& s, uint32_t capacity);

    BodyStore(const BodyStore&)            = delete;
    BodyStore& operator=(const BodyStore&) = delete;
    BodyStore(BodyStore&&)                 = default;
    BodyStore& operator=(BodyStore&&)      = default;

    // Add one body from host data. Returns the body ID (dense index).
    // Does not touch device memory; call upload() when done adding bodies.
    uint32_t add(const BodyParams& p);

    // Bulk-upload all bodies added since construction to device buffers.
    // Submissions are async on the queue used at construction; caller must
    // call stream.wait() before launching kernels that read body data.
    void upload();

    // Device-side SoA view for kernel capture. Returns mutable pointers so
    // kernels can update body state in-place. Valid until store is destroyed.
    BodyView view() noexcept;

    uint32_t count()    const noexcept { return count_; }
    uint32_t capacity() const noexcept { return cap_; }

    // Host-side body → shape index mapping; valid after add() calls.
    const std::vector<uint32_t>& body_shapes() const noexcept { return h_shape_; }

private:
    // Device buffers — one per SoA field, allocated at capacity.
    Buffer<float>    d_pos_x_, d_pos_y_, d_pos_z_;
    Buffer<float>    d_rot_w_, d_rot_x_, d_rot_y_, d_rot_z_;
    Buffer<float>    d_vel_x_, d_vel_y_, d_vel_z_;
    Buffer<float>    d_ang_x_, d_ang_y_, d_ang_z_;
    Buffer<float>    d_inv_mass_;
    Buffer<float>    d_iI_xx_, d_iI_yy_, d_iI_zz_;
    Buffer<float>    d_iI_xy_, d_iI_xz_, d_iI_yz_;
    Buffer<uint32_t> d_shape_, d_flags_;

    // Host shadow arrays — populated by add(), consumed by upload().
    std::vector<float>    h_pos_x_, h_pos_y_, h_pos_z_;
    std::vector<float>    h_rot_w_, h_rot_x_, h_rot_y_, h_rot_z_;
    std::vector<float>    h_vel_x_, h_vel_y_, h_vel_z_;
    std::vector<float>    h_ang_x_, h_ang_y_, h_ang_z_;
    std::vector<float>    h_inv_mass_;
    std::vector<float>    h_iI_xx_, h_iI_yy_, h_iI_zz_;
    std::vector<float>    h_iI_xy_, h_iI_xz_, h_iI_yz_;
    std::vector<uint32_t> h_shape_, h_flags_;

    uint32_t count_ = 0;
    uint32_t cap_   = 0;
};

} // namespace dyphur
