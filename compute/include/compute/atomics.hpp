#pragma once
#include <sycl/sycl.hpp>

namespace dyphur {

// Atomic add on a device pointer. Returns the old value.
// Call only from inside a kernel (device code).
template<typename T>
T atomic_add(T* ptr, T val) {
    sycl::atomic_ref<T,
        sycl::memory_order::relaxed,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space> ref(*ptr);
    return ref.fetch_add(val);
}

// Sequentially-consistent variant — stronger ordering guarantee.
template<typename T>
T atomic_add_seq(T* ptr, T val) {
    sycl::atomic_ref<T,
        sycl::memory_order::seq_cst,
        sycl::memory_scope::device,
        sycl::access::address_space::global_space> ref(*ptr);
    return ref.fetch_add(val);
}

} // namespace dyphur
