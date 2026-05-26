#pragma once
#include "vec3.hpp"
#include "quat.hpp"

namespace dyphur {

// Rigid transform: position + unit-quaternion rotation.
// Represents: p_world = rot.rotate(p_local) + pos
template<typename T>
struct Transform {
    Vec3<T> pos{};
    Quat<T> rot{Quat<T>::identity()};

    constexpr Transform() = default;
    constexpr Transform(Vec3<T> pos, Quat<T> rot) : pos(pos), rot(rot) {}

    // Transform a point (applies rotation then translation).
    constexpr Vec3<T> transform_point(const Vec3<T>& p) const {
        return rot.rotate(p) + pos;
    }

    // Transform a direction vector (rotation only, no translation).
    constexpr Vec3<T> transform_vector(const Vec3<T>& v) const {
        return rot.rotate(v);
    }

    // Inverse transform: from world to local.
    constexpr Transform inverse() const {
        Quat<T> inv_rot = rot.conjugate();
        return {inv_rot.rotate(-pos), inv_rot};
    }

    // Compose: this * o means apply o first, then this.
    constexpr Transform operator*(const Transform& o) const {
        return {transform_point(o.pos), rot * o.rot};
    }

    static constexpr Transform identity() {
        return {Vec3<T>::zero(), Quat<T>::identity()};
    }
};

using Transformf = Transform<float>;
using Transformd = Transform<double>;

} // namespace dyphur
