#pragma once
#include <cstdint>

namespace dyphur {

// Kernel-safe SoA view of the contact manifold.
// All pointer arrays indexed [0, *n).
struct ContactView {
    uint32_t* body_a;     // index of body A
    uint32_t* body_b;     // index of body B
    float*    pos_x;      // contact point in world space
    float*    pos_y;
    float*    pos_z;
    float*    norm_x;     // unit normal: from B toward A (push A out)
    float*    norm_y;
    float*    norm_z;
    float*    depth;      // penetration depth (> 0)
    uint32_t* n;          // device ptr: live contact count
    uint32_t  capacity;
};

} // namespace dyphur
