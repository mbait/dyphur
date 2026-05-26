#pragma once
#include "stream.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>

namespace dyphur {

// Parallel bitonic sort of (keys, values) by ascending key.
//
// Deterministic: the swap network is a fixed comparison network.
// Phase 0 constraint: n must be a power of 2.
// Complexity: O(n log^2 n) kernel work, O(log^2 n) sequential steps.
// Caller must call stream.wait() to synchronize before reading results.
template<typename Key, typename Value>
void sort_by_key(Stream& s, Key* keys, Value* values, size_t n) {
    if (n <= 1) return;

    // In-order queue: successive submissions execute in dependency order;
    // no explicit per-step barrier needed.
    auto& q = s.queue();

    for (size_t k = 2; k <= n; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                size_t i = id[0];
                size_t l = i ^ j;
                if (l > i) {
                    bool asc = !(i & k);
                    if ((asc  && keys[i] > keys[l]) ||
                        (!asc && keys[i] < keys[l])) {
                        Key   tk = keys[i];   keys[i]   = keys[l];   keys[l]   = tk;
                        Value tv = values[i]; values[i] = values[l]; values[l] = tv;
                    }
                }
            });
        }
    }
}

// Sort keys only (no associated values).
template<typename Key>
void sort(Stream& s, Key* keys, size_t n) {
    if (n <= 1) return;
    auto& q = s.queue();
    for (size_t k = 2; k <= n; k <<= 1) {
        for (size_t j = k >> 1; j > 0; j >>= 1) {
            q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
                size_t i = id[0];
                size_t l = i ^ j;
                if (l > i) {
                    bool asc = !(i & k);
                    if ((asc  && keys[i] > keys[l]) ||
                        (!asc && keys[i] < keys[l])) {
                        Key tk = keys[i]; keys[i] = keys[l]; keys[l] = tk;
                    }
                }
            });
        }
    }
}

} // namespace dyphur
