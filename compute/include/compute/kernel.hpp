#pragma once
#include "stream.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>

namespace dyphur {

// Launch a 1D kernel over [0, n). The callable receives a size_t index.
// Kernels must be trivially copyable; only device-safe operations inside.
template<typename F>
Event parallel_for(Stream& s, size_t n, F kernel) {
    return Event(s.queue().parallel_for(
        sycl::range<1>(n),
        [=](sycl::id<1> idx) { kernel(idx[0]); }));
}

} // namespace dyphur
