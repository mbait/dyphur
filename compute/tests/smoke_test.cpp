#include "compute/compute.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <numeric>

using namespace dyphur;

TEST_CASE("Device: enumerate", "[smoke]") {
    // Just picking any available device must not throw.
    auto dev = Device(sycl::device(sycl::default_selector_v));
    CHECK(!dev.name().empty());
}

TEST_CASE("Buffer: upload and download round-trip", "[smoke]") {
    auto dev = Device(sycl::device(sycl::default_selector_v));
    Stream s(dev.sycl_device());

    constexpr size_t n = 1024;
    std::vector<float> h_in(n), h_out(n, 0.f);
    std::iota(h_in.begin(), h_in.end(), 0.f);

    Buffer<float> buf(s, n);
    buf.upload(h_in.data()).wait();
    buf.download(h_out.data());

    for (size_t i = 0; i < n; ++i)
        REQUIRE(h_out[i] == h_in[i]);
}

TEST_CASE("parallel_for: saxpy", "[smoke]") {
    auto dev = Device(sycl::device(sycl::default_selector_v));
    Stream s(dev.sycl_device());

    constexpr size_t n = 1 << 16;
    constexpr float a = 2.5f;

    std::vector<float> h_x(n, 1.f), h_y(n, 3.f);

    Buffer<float> x(s, n), y(s, n);
    x.upload(h_x.data()).wait();
    y.upload(h_y.data()).wait();

    float* xp = x.data();
    float* yp = y.data();

    parallel_for(s, n, [=](size_t i) { yp[i] = a * xp[i] + yp[i]; });
    s.wait();

    std::vector<float> h_result(n);
    y.download(h_result.data());

    float expected = a * 1.f + 3.f;
    for (size_t i = 0; i < n; ++i)
        REQUIRE_THAT(h_result[i], Catch::Matchers::WithinULP(expected, 0));
}

TEST_CASE("reduce_sum: correctness", "[smoke]") {
    auto dev = Device(sycl::device(sycl::default_selector_v));
    Stream s(dev.sycl_device());

    constexpr size_t n = 1 << 14;
    std::vector<float> h(n, 1.f);

    Buffer<float> buf(s, n);
    buf.upload(h.data()).wait();

    float result = reduce_sum(s, buf.data(), n);
    REQUIRE_THAT(result, Catch::Matchers::WithinRel(float(n), 1e-5f));
}

TEST_CASE("sort_by_key: ascending order", "[smoke]") {
    auto dev = Device(sycl::device(sycl::default_selector_v));
    Stream s(dev.sycl_device());

    constexpr size_t n = 1 << 10;

    // Keys descending, values are original indices.
    std::vector<uint32_t> h_keys(n), h_vals(n);
    for (size_t i = 0; i < n; ++i) {
        h_keys[i] = static_cast<uint32_t>(n - 1 - i);
        h_vals[i] = static_cast<uint32_t>(i);
    }

    Buffer<uint32_t> keys(s, n), vals(s, n);
    keys.upload(h_keys.data()).wait();
    vals.upload(h_vals.data()).wait();

    sort_by_key(s, keys.data(), vals.data(), n);
    s.wait();

    keys.download(h_keys.data());
    vals.download(h_vals.data());

    for (size_t i = 0; i < n; ++i) {
        REQUIRE(h_keys[i] == static_cast<uint32_t>(i));
        REQUIRE(h_vals[i] == static_cast<uint32_t>(n - 1 - i));
    }
}
