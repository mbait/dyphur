#pragma once
#include "../../examples/common/scene_io.hpp"
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Shaders/Phong.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace dyphur::viz {

using namespace Magnum;
using namespace Magnum::Math::Literals;

struct BodyPose {
    float x, y, z;
    float qw, qx, qy, qz;
};

class Renderer {
public:
    // meshmap: optional per-body STL file paths (relative to exe directory).
    // Bodies with a meshmap entry are rendered with their actual mesh geometry.
    explicit Renderer(std::unordered_map<uint32_t, std::string> meshmap = {});

    void setViewProjection(const Matrix4& view, const Matrix4& proj);
    void draw(const SceneFileDesc& scene, const std::vector<BodyPose>& poses);

private:
    // Load binary STL and upload as a flat-shaded GL::Mesh (positions + normals).
    GL::Mesh load_stl_mesh(const std::string& path);

    GL::Mesh _box;
    GL::Mesh _sphere;
    Shaders::Phong _shader;
    Matrix4 _view, _proj;

    std::unordered_map<uint32_t, std::string> _meshmap;
    std::unordered_map<std::string, GL::Mesh> _mesh_cache;

    static Color3 bodyColor(uint32_t i, const SceneFileDesc& scene);
};

} // namespace dyphur::viz
