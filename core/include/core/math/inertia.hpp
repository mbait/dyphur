#pragma once
#include "vec3.hpp"
#include "mat3.hpp"
#include "spatial.hpp"

namespace dyphur {

// Rigid-body inertia expressed in a local body frame.
// Stores: scalar mass, center of mass (relative to frame origin), and
// the 3x3 rotational inertia tensor about the CoM.
template<typename T>
struct Inertia {
    T mass{T{1}};
    Vec3<T> com{};
    Mat3<T> I{Mat3<T>::identity()};

    constexpr Inertia() = default;
    constexpr Inertia(T mass, Vec3<T> com, Mat3<T> I) : mass(mass), com(com), I(I) {}

    // Construct for a point mass (zero rotational inertia).
    static constexpr Inertia point_mass(T m, Vec3<T> pos) {
        return {m, pos, {T{0},T{0},T{0}, T{0},T{0},T{0}, T{0},T{0},T{0}}};
    }

    // Apply parallel-axis theorem: shift inertia from CoM frame to origin.
    // Returns I_origin = I_com + m * (r.dot(r)*Id - r*r^T)
    constexpr Mat3<T> inertia_at_origin() const {
        T r2 = com.norm_sq();
        Mat3<T> outer = Mat3<T>::from_rows(com * com.x, com * com.y, com * com.z);
        Mat3<T> shift = Mat3<T>::identity() * (mass * r2) + outer * (-mass);
        return I + shift;
    }

    // 6x6 spatial inertia times a spatial velocity (returns a wrench).
    // M * v = [I_o * omega + m*(r × v_linear); m*(v_linear - r × omega)]
    // where r = com, omega = v.angular, v_linear = v.linear.
    constexpr SpatialVector<T> apply(const SpatialVector<T>& v) const {
        Vec3<T> omega = v.angular;
        Vec3<T> vlin  = v.linear;
        Vec3<T> rxom  = com.cross(omega);
        return {I * omega + com.cross(vlin * mass),
                (vlin - com.cross(omega)) * mass};
    }
};

using Inertiaf = Inertia<float>;
using Inertiad = Inertia<double>;

} // namespace dyphur
