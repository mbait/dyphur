#include "compute/compute.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <vector>

using namespace dyphur;

// Bit-cast a float to its raw uint32 representation.
static uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}

// Fill a host buffer with a deterministic pseudo-random sequence (LCG).
static std::vector<float> make_inputs(size_t n, uint32_t seed = 0xdeadbeef) {
    std::vector<float> v(n);
    uint32_t s = seed;
    for (size_t i = 0; i < n; ++i) {
        s = s * 1664525u + 1013904223u;
        // Map to (0, 1] to avoid NaN/Inf in reduction.
        v[i] = 1.f + float(s & 0xFFFF) * (1.f / 65536.f);
    }
    return v;
}

TEST_CASE("reduce_sum: bit-identical across 10 runs", "[determinism]") {
    constexpr size_t n = 1 << 20; // 1M floats
    constexpr int runs = 10;

    auto dev  = Device(sycl::device(sycl::default_selector_v));
    Stream s(dev.sycl_device());

    auto h_data = make_inputs(n);
    Buffer<float> buf(s, n);
    buf.upload(h_data.data()).wait();

    float first = reduce_sum(s, buf.data(), n);
    uint32_t golden = bits(first);

    for (int r = 1; r < runs; ++r) {
        float result = reduce_sum(s, buf.data(), n);
        INFO("Run " << r << ": expected bits=" << golden
             << " got bits=" << bits(result));
        REQUIRE(bits(result) == golden);
    }
}

TEST_CASE("sort: bit-identical output across 10 runs", "[determinism]") {
    constexpr size_t n = 1 << 12; // 4096 (power of 2)
    constexpr int runs = 10;

    auto dev = Device(sycl::device(sycl::default_selector_v));
    Stream s(dev.sycl_device());

    // Build a fixed input set.
    std::vector<uint32_t> h_keys_orig(n);
    uint32_t seed = 0xcafebabe;
    for (size_t i = 0; i < n; ++i) {
        seed = seed * 1664525u + 1013904223u;
        h_keys_orig[i] = seed;
    }

    // Sort once to get the golden output.
    std::vector<uint32_t> h_golden(n);
    {
        Buffer<uint32_t> keys(s, n), dummy(s, n);
        keys.upload(h_keys_orig.data()).wait();
        sort_by_key(s, keys.data(), dummy.data(), n);
        s.wait();
        keys.download(h_golden.data());
    }

    // Re-sort from the same input, check bit-identical each time.
    for (int r = 0; r < runs; ++r) {
        Buffer<uint32_t> keys(s, n), dummy(s, n);
        keys.upload(h_keys_orig.data()).wait();
        sort_by_key(s, keys.data(), dummy.data(), n);
        s.wait();

        std::vector<uint32_t> h_result(n);
        keys.download(h_result.data());

        for (size_t i = 0; i < n; ++i) {
            INFO("Run " << r << " position " << i);
            REQUIRE(h_result[i] == h_golden[i]);
        }
    }
}
