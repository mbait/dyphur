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
