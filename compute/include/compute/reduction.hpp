#pragma once
#include "stream.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>
#include <vector>

namespace dyphur {

namespace detail {
    inline constexpr size_t kReduceWgSize = 256;
} // namespace detail

// Deterministic parallel sum reduction.
//
// Two-pass approach for bit-identical results across runs on the same
// hardware + build:
//   Pass 1: tree-reduction within each work group (fixed order, local memory).
//   Pass 2: sequential host-side merge of partial sums (fixed order).
//
// Allocates a temporary device buffer internally; synchronizes before return.
template<typename T>
T reduce_sum(Stream& s, const T* d_data, size_t n) {
    if (n == 0) return T{0};

    constexpr size_t wg = detail::kReduceWgSize;
    size_t num_groups = (n + wg - 1) / wg;

    T* d_partial = sycl::malloc_device<T>(num_groups, s.queue());

    s.queue().submit([&](sycl::handler& h) {
        sycl::local_accessor<T, 1> scratch(sycl::range<1>(wg), h);

        h.parallel_for(
            sycl::nd_range<1>(num_groups * wg, wg),
            [=](sycl::nd_item<1> item) {
                size_t gid = item.get_global_id(0);
                size_t lid = item.get_local_id(0);

                scratch[lid] = (gid < n) ? d_data[gid] : T{0};
                sycl::group_barrier(item.get_group());

                for (size_t stride = wg / 2; stride > 0; stride >>= 1) {
                    if (lid < stride)
                        scratch[lid] += scratch[lid + stride];
                    sycl::group_barrier(item.get_group());
                }

                if (lid == 0)
                    d_partial[item.get_group(0)] = scratch[0];
            });
    });

    // Memcpy implicitly follows the kernel in an in-order queue.
    std::vector<T> h_partial(num_groups);
    s.queue().memcpy(h_partial.data(), d_partial, num_groups * sizeof(T)).wait();
    sycl::free(d_partial, s.queue());

    T result = T{0};
    for (size_t i = 0; i < num_groups; ++i)
        result += h_partial[i];
    return result;
}

} // namespace dyphur
