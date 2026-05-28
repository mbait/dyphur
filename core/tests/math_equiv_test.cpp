#include "core/math/math.hpp"
#include "compute/compute.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

using namespace dyphur;


// ── Vec3 cross product: host vs device ───────────────────────────────────────

TEST_CASE("Vec3: host-device cross product equivalence", "[smoke]") {
    auto dev = Device(sycl::device(sycl::default_selector_v));
    Stream s(dev.sycl_device());

    constexpr size_t n = 1024;

    // Input vectors as SoA flat arrays.
    std::vector<float> h_ax(n), h_ay(n), h_az(n);
    std::vector<float> h_bx(n), h_by(n), h_bz(n);
    for (size_t i = 0; i < n; ++i) {
        h_ax[i] = float(i) + 0.1f; h_ay[i] = float(i) + 0.2f; h_az[i] = float(i) + 0.3f;
        h_bx[i] = float(n - i);    h_by[i] = float(n - i) * 2.f; h_bz[i] = 1.f;
    }

    // Host-side results.
    std::vector<float> h_rx(n), h_ry(n), h_rz(n);
    for (size_t i = 0; i < n; ++i) {
        Vec3f a{h_ax[i], h_ay[i], h_az[i]};
        Vec3f b{h_bx[i], h_by[i], h_bz[i]};
        Vec3f r = a.cross(b);
        h_rx[i] = r.x; h_ry[i] = r.y; h_rz[i] = r.z;
    }

    // Device-side results.
    Buffer<float> d_ax(s,n), d_ay(s,n), d_az(s,n);
    Buffer<float> d_bx(s,n), d_by(s,n), d_bz(s,n);
    Buffer<float> d_rx(s,n), d_ry(s,n), d_rz(s,n);

    d_ax.upload(h_ax.data()).wait(); d_ay.upload(h_ay.data()).wait(); d_az.upload(h_az.data()).wait();
    d_bx.upload(h_bx.data()).wait(); d_by.upload(h_by.data()).wait(); d_bz.upload(h_bz.data()).wait();

    float* pax=d_ax.data(), *pay=d_ay.data(), *paz=d_az.data();
    float* pbx=d_bx.data(), *pby=d_by.data(), *pbz=d_bz.data();
    float* prx=d_rx.data(), *pry=d_ry.data(), *prz=d_rz.data();

    parallel_for(s, n, [=](size_t i) {
        Vec3f a{pax[i], pay[i], paz[i]};
        Vec3f b{pbx[i], pby[i], pbz[i]};
        Vec3f r = a.cross(b);
        prx[i] = r.x; pry[i] = r.y; prz[i] = r.z;
    });
    s.wait();

    std::vector<float> d_out_x(n), d_out_y(n), d_out_z(n);
    d_rx.download(d_out_x.data());
    d_ry.download(d_out_y.data());
    d_rz.download(d_out_z.data());

    for (size_t i = 0; i < n; ++i) {
        INFO("element " << i);
        REQUIRE_THAT(d_out_x[i], Catch::Matchers::WithinULP(h_rx[i], 1));
        REQUIRE_THAT(d_out_y[i], Catch::Matchers::WithinULP(h_ry[i], 1));
        REQUIRE_THAT(d_out_z[i], Catch::Matchers::WithinULP(h_rz[i], 1));
    }
}

// ── Quat rotation: host vs device ────────────────────────────────────────────

TEST_CASE("Quat: host-device rotate equivalence", "[smoke]") {
    auto dev = Device(sycl::device(sycl::default_selector_v));
    Stream s(dev.sycl_device());

    constexpr size_t n = 512;

    // Fixed quaternion: 45° around (1,1,0)/sqrt(2).
    Vec3f axis = Vec3f{1.f, 1.f, 0.f}.normalized();
    Quatf q    = Quatf::from_axis_angle(axis, float(M_PI) / 4.f);

    std::vector<float> h_vx(n), h_vy(n), h_vz(n);
    for (size_t i = 0; i < n; ++i) {
        h_vx[i] = float(i); h_vy[i] = float(i) * 0.5f; h_vz[i] = 1.f;
    }

    // Host results.
    std::vector<float> h_rx(n), h_ry(n), h_rz(n);
    for (size_t i = 0; i < n; ++i) {
        Vec3f r = q.rotate({h_vx[i], h_vy[i], h_vz[i]});
        h_rx[i] = r.x; h_ry[i] = r.y; h_rz[i] = r.z;
    }

    // Device results.
    Buffer<float> d_vx(s,n), d_vy(s,n), d_vz(s,n);
    Buffer<float> d_rx(s,n), d_ry(s,n), d_rz(s,n);
    d_vx.upload(h_vx.data()).wait(); d_vy.upload(h_vy.data()).wait(); d_vz.upload(h_vz.data()).wait();

    float* pvx=d_vx.data(), *pvy=d_vy.data(), *pvz=d_vz.data();
    float* prx=d_rx.data(), *pry=d_ry.data(), *prz=d_rz.data();

    parallel_for(s, n, [=](size_t i) {
        Vec3f r = q.rotate({pvx[i], pvy[i], pvz[i]});
        prx[i] = r.x; pry[i] = r.y; prz[i] = r.z;
    });
    s.wait();

    std::vector<float> ox(n), oy(n), oz(n);
    d_rx.download(ox.data()); d_ry.download(oy.data()); d_rz.download(oz.data());

    for (size_t i = 0; i < n; ++i) {
        INFO("element " << i);
        REQUIRE_THAT(ox[i], Catch::Matchers::WithinULP(h_rx[i], 1));
        REQUIRE_THAT(oy[i], Catch::Matchers::WithinULP(h_ry[i], 1));
        REQUIRE_THAT(oz[i], Catch::Matchers::WithinULP(h_rz[i], 1));
    }
}
