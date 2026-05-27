#pragma once
#include "scene_desc.hpp"
#include "convex_decomp.hpp"
#include <string>

namespace dyphur {

struct SdfLoadParams {
    // V-HACD parameters used when mesh geometry is assigned to a dynamic body.
    DecompParams decomp;
    // When true, mesh geometry on static bodies is kept as TriangleMesh (BVH).
    // When false, static mesh bodies also go through V-HACD decomposition.
    bool static_mesh_as_trimesh = true;
    // Scale factor applied to all positions and dimensions (SDF default: metres).
    float scale = 1.f;
};

// Load an SDF world or model file.
// - Box / sphere / cylinder geometry → Box or Sphere ShapeType.
//   Cylinder is approximated as a Box (half_x = half_z = radius, half_y = half_length).
// - Mesh geometry on dynamic bodies → V-HACD convex hull decomposition.
// - Mesh geometry on static bodies  → TriangleMesh (BVH), unless params.static_mesh_as_trimesh=false.
// - Joints (fixed/revolute/prismatic/ball) → JointParams with body indices resolved.
//
// Throws std::runtime_error on file-not-found or parse failure.
SceneDesc load_sdf(const std::string& path, const SdfLoadParams& params = {});

} // namespace dyphur
