#pragma once
#include "stream.hpp"
#include <sycl/sycl.hpp>
#include <string>

namespace dyphur {

class Device {
public:
    explicit Device(sycl::device dev) : dev_(std::move(dev)) {}

    static Device default_gpu() { return Device(sycl::device(sycl::gpu_selector_v)); }
    static Device default_cpu() { return Device(sycl::device(sycl::cpu_selector_v)); }

    // Preferred device for this build.  Only selects a GPU when GPU kernels were
    // actually compiled (DYPHUR_BACKEND_IS_{CUDA,HIP,L0}); an OMP-only build returns
    // the CPU device, because AdaptiveCpp would otherwise hand back a physical GPU
    // with no kernel launcher and abort at first launch.  May still throw on a GPU
    // build with no usable device, so callers should keep a try/catch → CPU fallback.
    static Device preferred() {
#if defined(DYPHUR_BACKEND_IS_CUDA) || defined(DYPHUR_BACKEND_IS_HIP) || defined(DYPHUR_BACKEND_IS_L0)
        return default_gpu();
#else
        return default_cpu();
#endif
    }

    std::string name() const { return dev_.get_info<sycl::info::device::name>(); }
    bool is_gpu()      const { return dev_.is_gpu(); }
    bool is_cpu()      const { return dev_.is_cpu(); }

    size_t max_compute_units() const {
        return dev_.get_info<sycl::info::device::max_compute_units>();
    }
    size_t global_mem_bytes() const {
        return dev_.get_info<sycl::info::device::global_mem_size>();
    }

    Stream make_stream() const { return Stream(dev_); }

    const sycl::device& sycl_device() const { return dev_; }

private:
    sycl::device dev_;
};

} // namespace dyphur
