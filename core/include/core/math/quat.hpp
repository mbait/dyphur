#pragma once
#include "vec3.hpp"
#include "mat3.hpp"
#include <cmath>

namespace dyphur {

// Unit quaternion: w + xi + yj + zk.
// Convention: q = (w, xyz) where w is the scalar part.
template<typename T>
struct Quat {
    T w{T{1}};
    Vec3<T> xyz{};

    constexpr Quat() = default;
    constexpr Quat(T w, Vec3<T> xyz) : w(w), xyz(xyz) {}
    constexpr Quat(T w, T x, T y, T z) : w(w), xyz{x, y, z} {}

    // Angle-axis constructor: axis must be unit-length, angle in radians.
    static Quat from_axis_angle(const Vec3<T>& axis, T angle) {
        T half = angle * T{0.5};
        return {std::cos(half), axis * std::sin(half)};
    }

    // Quaternion product: apply this rotation first, then o.
    constexpr Quat operator*(const Quat& o) const {
        return {w * o.w - xyz.dot(o.xyz),
                xyz.cross(o.xyz) + o.xyz * w + xyz * o.w};
    }

    constexpr Quat conjugate()  const { return {w, -xyz}; }
    constexpr Quat operator-()  const { return {-w, -xyz}; }

    constexpr T norm_sq() const { return w * w + xyz.norm_sq(); }
    T norm()              const { return std::sqrt(norm_sq()); }
    Quat normalized()     const { T n = norm(); return {w / n, xyz / n}; }

    // Rotate a vector: v' = q * [0,v] * q†  (Rodrigues, no full quat mult).
    constexpr Vec3<T> rotate(const Vec3<T>& v) const {
        Vec3<T> t = xyz.cross(v) * T{2};
        return v + t * w + xyz.cross(t);
    }

    // Convert to rotation matrix.
    constexpr Mat3<T> to_matrix() const {
        T x = xyz.x, y = xyz.y, z = xyz.z;
        return {T{1} - T{2}*(y*y + z*z),  T{2}*(x*y - w*z),       T{2}*(x*z + w*y),
                T{2}*(x*y + w*z),          T{1} - T{2}*(x*x + z*z), T{2}*(y*z - w*x),
                T{2}*(x*z - w*y),          T{2}*(y*z + w*x),        T{1} - T{2}*(x*x + y*y)};
    }

    static constexpr Quat identity() { return {T{1}, Vec3<T>::zero()}; }
};

using Quatf = Quat<float>;
using Quatd = Quat<double>;

} // namespace dyphur
