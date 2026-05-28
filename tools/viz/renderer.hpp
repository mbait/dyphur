#pragma once
#include "../../examples/common/scene_io.hpp"
#include <Magnum/GL/Mesh.h>
#include <Magnum/Math/Color.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Shaders/Phong.h>
#include <vector>

namespace dyphur::viz {

using namespace Magnum;
using namespace Magnum::Math::Literals;

// Per-body pose read from a trajectory frame.
struct BodyPose {
    float x, y, z;
    float qw, qx, qy, qz;
};

// Holds GL resources shared between snapshot and replay modes.
// Must be constructed after a GL context is current.
class Renderer {
public:
    Renderer();

    void setViewProjection(const Matrix4& view, const Matrix4& proj);

    // Render one frame. scene: static shape catalog. poses: one per body.
    void draw(const SceneFileDesc& scene, const std::vector<BodyPose>& poses);

private:
    GL::Mesh _box;
    GL::Mesh _sphere;
    Shaders::Phong _shader;
    Matrix4 _view, _proj;

    // Assign a stable colour to body i.
    static Color3 bodyColor(uint32_t i, const SceneFileDesc& scene);
};

} // namespace dyphur::viz
