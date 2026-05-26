#pragma once
#include <cstdint>

namespace dyphur {

enum class ShapeType : uint32_t { Box = 0, Sphere = 1 };

struct ShapeParams {
    ShapeType type   = ShapeType::Box;
    float     half_x = 0.5f;  // box: X half-extent; sphere: radius
    float     half_y = 0.5f;  // box: Y half-extent; unused for sphere
    float     half_z = 0.5f;  // box: Z half-extent; unused for sphere
};

// Kernel-safe SoA view of the shape catalog (read-only in kernels).
struct ShapeView {
    const uint32_t* type;
    const float*    half_x;
    const float*    half_y;
    const float*    half_z;
    uint32_t        n;
};

} // namespace dyphur
