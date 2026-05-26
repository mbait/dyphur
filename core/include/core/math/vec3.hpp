#pragma once
#include <cmath>
#include <cstddef>
#include <type_traits>

namespace dyphur {

template<typename T>
struct Vec3 {
    T x{}, y{}, z{};

    constexpr Vec3() = default;
    constexpr Vec3(T x, T y, T z) : x(x), y(y), z(z) {}

    constexpr Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    constexpr Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    constexpr Vec3 operator-()              const { return {-x, -y, -z}; }
    constexpr Vec3 operator*(T s)           const { return {x * s, y * s, z * s}; }
    constexpr Vec3 operator/(T s)           const { return {x / s, y / s, z / s}; }

    constexpr Vec3& operator+=(const Vec3& o) { x += o.x; y += o.y; z += o.z; return *this; }
    constexpr Vec3& operator-=(const Vec3& o) { x -= o.x; y -= o.y; z -= o.z; return *this; }
    constexpr Vec3& operator*=(T s)           { x *= s;   y *= s;   z *= s;   return *this; }

    constexpr T dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }

    constexpr Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y,
                z * o.x - x * o.z,
                x * o.y - y * o.x};
    }

    constexpr T norm_sq() const { return dot(*this); }
    T norm()              const { return std::sqrt(norm_sq()); }
    Vec3 normalized()     const { return *this / norm(); }

    constexpr bool operator==(const Vec3& o) const {
        return x == o.x && y == o.y && z == o.z;
    }

    constexpr T& operator[](size_t i)       { return (&x)[i]; }
    constexpr T  operator[](size_t i) const { return (&x)[i]; }

    static constexpr Vec3 zero()   { return {T{0}, T{0}, T{0}}; }
    static constexpr Vec3 ones()   { return {T{1}, T{1}, T{1}}; }
    static constexpr Vec3 unit_x() { return {T{1}, T{0}, T{0}}; }
    static constexpr Vec3 unit_y() { return {T{0}, T{1}, T{0}}; }
    static constexpr Vec3 unit_z() { return {T{0}, T{0}, T{1}}; }
};

template<typename T>
constexpr Vec3<T> operator*(T s, const Vec3<T>& v) { return v * s; }

using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;

} // namespace dyphur
