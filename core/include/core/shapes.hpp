#pragma once
#include <cstdint>

namespace dyphur {

enum class ShapeType : uint32_t {
    Box         = 0,
    Sphere      = 1,
    ConvexHull  = 2,  // ext_id → ConvexHullStore index
    TriangleMesh = 3, // ext_id → MeshBvhStore index (static bodies only)
};

struct ShapeParams {
    ShapeType type   = ShapeType::Box;
    float     half_x = 0.5f;  // box: X half-extent; sphere: radius; others: unused
    float     half_y = 0.5f;  // box: Y half-extent; unused otherwise
    float     half_z = 0.5f;  // box: Z half-extent; unused otherwise
    uint32_t  ext_id = 0;     // ConvexHull/TriangleMesh: catalog index
};

// Kernel-safe SoA view of the shape catalog (read-only in kernels).
struct ShapeView {
    const uint32_t* type;
    const float*    half_x;
    const float*    half_y;
    const float*    half_z;
    const uint32_t* ext_id;
    uint32_t        n;
};

} // namespace dyphur
