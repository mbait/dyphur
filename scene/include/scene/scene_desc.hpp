#pragma once
#include <core/body.hpp>
#include <core/shapes.hpp>
#include <core/articulation.hpp>
#include <string>
#include <vector>
#include <cstdint>

namespace dyphur {

// Raw triangle mesh data (CPU-side, before convex decomp or BVH build).
struct VertexBuffer {
    std::vector<float>    x, y, z;            // vertex positions
    std::vector<uint32_t> idx_a, idx_b, idx_c; // triangle indices (empty for convex hulls)
};

// One body from the SDF world/model.
// shape.type == ConvexHull  → hull_idx indexes into SceneDesc::hulls
// shape.type == TriangleMesh → mesh_idx indexes into SceneDesc::meshes
struct BodyDesc {
    BodyParams  body;
    ShapeParams shape;
    std::string name;
    // ext_id in shape is set by the loader to these indices.
    // hull_idx / mesh_idx are the local SceneDesc indices before any catalog upload.
    uint32_t    hull_idx = UINT32_MAX;
    uint32_t    mesh_idx = UINT32_MAX;
};

// One joint resolved to body indices in this SceneDesc.
struct JointDesc {
    JointParams joint;
    std::string parent_name;
    std::string child_name;
};

// Flat output of the SDF loader.
// Coordinate frame: SDF native (z-up). Callers must set gravity = {0,0,-9.81f}.
struct SceneDesc {
    std::vector<BodyDesc>    bodies;
    std::vector<JointDesc>   joints; // parent/child resolved to body indices via name lookup
    std::vector<VertexBuffer> hulls; // ConvexHull shape data
    std::vector<VertexBuffer> meshes; // TriangleMesh shape data
};

} // namespace dyphur
