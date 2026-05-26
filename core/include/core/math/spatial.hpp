#pragma once
#include "vec3.hpp"

namespace dyphur {

// 6D spatial vector: (angular, linear).
// Used for both velocities (omega, v) and wrenches (torque, force).
template<typename T>
struct SpatialVector {
    Vec3<T> angular{};
    Vec3<T> linear{};

    constexpr SpatialVector() = default;
    constexpr SpatialVector(Vec3<T> angular, Vec3<T> linear)
        : angular(angular), linear(linear) {}

    constexpr SpatialVector operator+(const SpatialVector& o) const {
        return {angular + o.angular, linear + o.linear};
    }
    constexpr SpatialVector operator-(const SpatialVector& o) const {
        return {angular - o.angular, linear - o.linear};
    }
    constexpr SpatialVector operator*(T s) const {
        return {angular * s, linear * s};
    }
    constexpr SpatialVector& operator+=(const SpatialVector& o) {
        angular += o.angular; linear += o.linear; return *this;
    }

    // Spatial dot product: <v1, v2> = v1.angular.dot(v2.linear) + v1.linear.dot(v2.angular)
    constexpr T dot(const SpatialVector& o) const {
        return angular.dot(o.linear) + linear.dot(o.angular);
    }

    static constexpr SpatialVector zero() {
        return {Vec3<T>::zero(), Vec3<T>::zero()};
    }
};

template<typename T>
constexpr SpatialVector<T> operator*(T s, const SpatialVector<T>& v) { return v * s; }

using SpatialVectorf = SpatialVector<float>;
using SpatialVectord = SpatialVector<double>;

} // namespace dyphur
