#pragma once
#include "stream.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>

namespace dyphur {

// Parallel bitonic sort of (keys, values) by ascending key.
//
// Deterministic: the swap network is a fixed comparison network.
// n must be a power of 2.
// For n ≤ 1024: single kernel launch using work-group local memory — all
// bitonic passes execute on-chip without PCIe round-trips between steps.
// For n > 1024: O(log²n) kernel submissions (in-order queue serialises them).
// Caller must call stream.wait() to synchronize before reading results.
template<typename Key, typename Value>
void sort_by_key(Stream& s, Key* keys, Value* values, size_t n) {
    if (n <= 1) return;

    auto& q = s.queue();

    if (n <= 1024) {
        // Single kernel: load into local memory, run all passes with barriers,
        // write back. Reduces GPU kernel launches from O(log²n) to 1.
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<Key,   1> lk(sycl::range<1>(n), h);
            sycl::local_accessor<Value, 1> lv(sycl::range<1>(n), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(n)),
                [=](sycl::nd_item<1> it) {
                    size_t i = it.get_local_id(0);
                    lk[i] = keys[i];
                    lv[i] = values[i];
                    it.barrier(sycl::access::fence_space::local_space);
                    for (size_t k = 2; k <= n; k <<= 1) {
                        for (size_t j = k >> 1; j > 0; j >>= 1) {
                            size_t l = i ^ j;
                            if (l > i) {
                                bool asc = !(i & k);
                                if ((asc  && lk[i] > lk[l]) ||
                                    (!asc && lk[i] < lk[l])) {
                                    Key   tk = lk[i]; lk[i] = lk[l]; lk[l] = tk;
                                    Value tv = lv[i]; lv[i] = lv[l]; lv[l] = tv;
                                }
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                    }
                    keys[i]   = lk[i];
                    values[i] = lv[i];
                });
        });
        return;
    }

    // Multi-launch fallback for n > 1024.
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

    if (n <= 1024) {
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<Key, 1> lk(sycl::range<1>(n), h);
            h.parallel_for(
                sycl::nd_range<1>(sycl::range<1>(n), sycl::range<1>(n)),
                [=](sycl::nd_item<1> it) {
                    size_t i = it.get_local_id(0);
                    lk[i] = keys[i];
                    it.barrier(sycl::access::fence_space::local_space);
                    for (size_t k = 2; k <= n; k <<= 1) {
                        for (size_t j = k >> 1; j > 0; j >>= 1) {
                            size_t l = i ^ j;
                            if (l > i) {
                                bool asc = !(i & k);
                                if ((asc  && lk[i] > lk[l]) ||
                                    (!asc && lk[i] < lk[l])) {
                                    Key tk = lk[i]; lk[i] = lk[l]; lk[l] = tk;
                                }
                            }
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                    }
                    keys[i] = lk[i];
                });
        });
        return;
    }

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
