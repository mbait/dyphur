#pragma once
#include "scene_desc.hpp"
#include <cstdint>

namespace dyphur {

struct DecompParams {
    uint32_t max_hulls              = 16;
    uint32_t resolution             = 100000;
    uint32_t max_verts_per_hull     = 64;
    double   min_volume_percent_error = 1.0; // V-HACD v4: stop if voxel volume within X% of hull
};

// Decompose a triangle mesh into a set of approximately-convex parts using V-HACD.
// Returns one VertexBuffer per hull (idx_a/idx_b/idx_c are empty — hull vertices only).
// Throws std::runtime_error if the decomposition fails.
std::vector<VertexBuffer> decompose_vhacd(const VertexBuffer& mesh,
                                          const DecompParams& params = {});

} // namespace dyphur
