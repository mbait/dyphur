#pragma once
#include <cstdint>

namespace dyphur {

// Axis-aligned bounding box. Kernel-safe: only floats, constexpr ops.
struct AABB {
    float min_x, min_y, min_z;
    float max_x, max_y, max_z;

    static constexpr AABB empty() {
        return { 1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f };
    }

    static constexpr AABB merge(const AABB& a, const AABB& b) {
        return {
            a.min_x < b.min_x ? a.min_x : b.min_x,
            a.min_y < b.min_y ? a.min_y : b.min_y,
            a.min_z < b.min_z ? a.min_z : b.min_z,
            a.max_x > b.max_x ? a.max_x : b.max_x,
            a.max_y > b.max_y ? a.max_y : b.max_y,
            a.max_z > b.max_z ? a.max_z : b.max_z,
        };
    }

    constexpr bool overlaps(const AABB& o) const {
        return min_x <= o.max_x && max_x >= o.min_x &&
               min_y <= o.max_y && max_y >= o.min_y &&
               min_z <= o.max_z && max_z >= o.min_z;
    }
};

} // namespace dyphur
