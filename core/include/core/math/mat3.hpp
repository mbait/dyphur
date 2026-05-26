#pragma once
#include "vec3.hpp"
#include <array>
#include <cmath>

namespace dyphur {

// Row-major 3x3 matrix. m(row, col) = data_[row * 3 + col].
template<typename T>
struct Mat3 {
    std::array<T, 9> data_{};

    constexpr Mat3() = default;

    // Row-major init: { row0.x, row0.y, row0.z, row1.x, ... }
    constexpr Mat3(T m00, T m01, T m02,
                   T m10, T m11, T m12,
                   T m20, T m21, T m22)
        : data_{m00, m01, m02, m10, m11, m12, m20, m21, m22} {}

    constexpr T& operator()(size_t r, size_t c)       { return data_[r * 3 + c]; }
    constexpr T  operator()(size_t r, size_t c) const { return data_[r * 3 + c]; }

    constexpr Vec3<T> row(size_t r) const {
        return {data_[r * 3], data_[r * 3 + 1], data_[r * 3 + 2]};
    }
    constexpr Vec3<T> col(size_t c) const {
        return {data_[c], data_[3 + c], data_[6 + c]};
    }

    constexpr Mat3 transpose() const {
        return {data_[0], data_[3], data_[6],
                data_[1], data_[4], data_[7],
                data_[2], data_[5], data_[8]};
    }

    constexpr Vec3<T> operator*(const Vec3<T>& v) const {
        return {row(0).dot(v), row(1).dot(v), row(2).dot(v)};
    }

    constexpr Mat3 operator*(const Mat3& o) const {
        Mat3 r;
        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j)
                r(i, j) = row(i).dot(o.col(j));
        return r;
    }

    constexpr Mat3 operator+(const Mat3& o) const {
        Mat3 r;
        for (size_t i = 0; i < 9; ++i) r.data_[i] = data_[i] + o.data_[i];
        return r;
    }

    constexpr Mat3 operator*(T s) const {
        Mat3 r;
        for (size_t i = 0; i < 9; ++i) r.data_[i] = data_[i] * s;
        return r;
    }

    constexpr T det() const {
        return data_[0] * (data_[4] * data_[8] - data_[5] * data_[7])
             - data_[1] * (data_[3] * data_[8] - data_[5] * data_[6])
             + data_[2] * (data_[3] * data_[7] - data_[4] * data_[6]);
    }

    // Inverse via adjugate (valid only when det != 0).
    constexpr Mat3 inverse() const {
        T d = det();
        return Mat3{ (data_[4]*data_[8] - data_[5]*data_[7]) / d,
                    -(data_[1]*data_[8] - data_[2]*data_[7]) / d,
                     (data_[1]*data_[5] - data_[2]*data_[4]) / d,
                    -(data_[3]*data_[8] - data_[5]*data_[6]) / d,
                     (data_[0]*data_[8] - data_[2]*data_[6]) / d,
                    -(data_[0]*data_[5] - data_[2]*data_[3]) / d,
                     (data_[3]*data_[7] - data_[4]*data_[6]) / d,
                    -(data_[0]*data_[7] - data_[1]*data_[6]) / d,
                     (data_[0]*data_[4] - data_[1]*data_[3]) / d };
    }

    static constexpr Mat3 identity() {
        return {T{1}, T{0}, T{0},
                T{0}, T{1}, T{0},
                T{0}, T{0}, T{1}};
    }

    // Skew-symmetric matrix [v]× such that [v]× w = v.cross(w).
    static constexpr Mat3 skew(const Vec3<T>& v) {
        return { T{0}, -v.z,  v.y,
                  v.z, T{0}, -v.x,
                 -v.y,  v.x, T{0}};
    }

    static constexpr Mat3 from_rows(const Vec3<T>& r0, const Vec3<T>& r1, const Vec3<T>& r2) {
        return {r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z};
    }

    static constexpr Mat3 from_cols(const Vec3<T>& c0, const Vec3<T>& c1, const Vec3<T>& c2) {
        return {c0.x, c1.x, c2.x, c0.y, c1.y, c2.y, c0.z, c1.z, c2.z};
    }
};

template<typename T>
constexpr Mat3<T> operator*(T s, const Mat3<T>& m) { return m * s; }

using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;

} // namespace dyphur
