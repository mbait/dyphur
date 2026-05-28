#include "renderer.hpp"
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/UVSphere.h>
#include <Magnum/Trade/MeshData.h>

namespace dyphur::viz {

using namespace Magnum;
using namespace Magnum::Math::Literals;

Renderer::Renderer()
    : _box{MeshTools::compile(Primitives::cubeSolid())}
    , _sphere{MeshTools::compile(Primitives::uvSphereSolid(8, 16))}
    , _shader{Shaders::Phong::Flags{}, 1}
{
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

    _shader.setAmbientColor(0x333333_rgbf)
           .setSpecularColor(0xffffff_rgbf)
           .setShininess(60.f);
}

void Renderer::setViewProjection(const Matrix4& view, const Matrix4& proj) {
    _view = view;
    _proj = proj;
}

void Renderer::draw(const SceneFileDesc& scene, const std::vector<BodyPose>& poses) {
    // Light position in camera space (Phong shader expects camera-space positions).
    Vector3 light_cam = (_view * Vector4{10.f, 20.f, 10.f, 1.f}).xyz();
    _shader.setLightPositions({{light_cam}});

    for (uint32_t i = 0; i < static_cast<uint32_t>(poses.size()); ++i) {
        const auto& p = poses[i];
        uint32_t si = scene.body_shape_idx[i];
        const auto& sp = scene.shapes[si];

        // Body-to-world transform.
        Quaternion q{{p.qx, p.qy, p.qz}, p.qw};
        Matrix4 body_tf = Matrix4::from(q.normalized().toMatrix(), {p.x, p.y, p.z});

        // Scale: Magnum's cubeSolid spans [-1,1]; uvSphereSolid radius=1.
        // Multiply by half-extents to get the right physical size.
        Vector3 scale{sp.half_x, sp.half_y, sp.half_z};
        if (scale == Vector3{}) scale = {0.1f, 0.1f, 0.1f}; // fallback for zero-size

        Matrix4 model = body_tf * Matrix4::scaling(scale);
        Matrix4 mv    = _view * model;

        _shader.setDiffuseColor(bodyColor(i, scene))
               .setTransformationMatrix(mv)
               .setNormalMatrix(mv.normalMatrix())
               .setProjectionMatrix(_proj);

        bool is_sphere = (sp.type == ShapeType::Sphere);
        _shader.draw(is_sphere ? _sphere : _box);
    }
}

Color3 Renderer::bodyColor(uint32_t i, const SceneFileDesc& scene) {
    // Palette of 8 saturated colours; static bodies are gray.
    static const Color3 palette[] = {
        0x4466cc_rgbf, 0x44cc66_rgbf, 0xcc6644_rgbf, 0xccaa44_rgbf,
        0x44cccc_rgbf, 0xcc44cc_rgbf, 0xaacc44_rgbf, 0x4488ff_rgbf,
    };
    // Use a heuristic: if the shape index is the last shape and there's only
    // one such body, it's likely the ground/static body.  Otherwise cycle.
    (void)scene; // reserved for future static-body detection
    return palette[i % 8];
}

} // namespace dyphur::viz
