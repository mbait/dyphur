#include "renderer.hpp"
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Quaternion.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/Primitives/Cube.h>
#include <Magnum/Primitives/UVSphere.h>
#include <Magnum/Trade/MeshData.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace dyphur::viz {

using namespace Magnum;
using namespace Magnum::Math::Literals;

Renderer::Renderer(std::unordered_map<uint32_t, std::string> meshmap)
    : _box{MeshTools::compile(Primitives::cubeSolid())}
    , _sphere{MeshTools::compile(Primitives::uvSphereSolid(8, 16))}
    , _shader{Shaders::Phong::Flags{}, 1}
    , _meshmap{std::move(meshmap)}
{
    GL::Renderer::enable(GL::Renderer::Feature::DepthTest);
    GL::Renderer::enable(GL::Renderer::Feature::FaceCulling);

    _shader.setAmbientColor(0x333333_rgbf)
           .setSpecularColor(0xffffff_rgbf)
           .setShininess(60.f);
}

GL::Mesh Renderer::load_stl_mesh(const std::string& path)
{
    // Binary STL: 80-byte header, uint32 n_tri, then per-triangle:
    //   float[3] normal, float[3]*3 vertices, uint16 attribute.
    // We upload one position + one normal per vertex (flat shading: face normal repeated).
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "renderer: cannot open STL %s — using box\n", path.c_str());
        return MeshTools::compile(Primitives::cubeSolid());
    }

    f.seekg(80);
    uint32_t n_tri = 0;
    f.read(reinterpret_cast<char*>(&n_tri), 4);

    struct Vertex { float px, py, pz, nx, ny, nz; };
    std::vector<Vertex> verts;
    verts.reserve(n_tri * 3);

    for (uint32_t t = 0; t < n_tri; ++t) {
        float buf[12]; // normal(3) + v0(3) + v1(3) + v2(3)
        f.read(reinterpret_cast<char*>(buf), 48);
        f.ignore(2);
        float nx = buf[0], ny = buf[1], nz = buf[2];
        for (int v = 0; v < 3; ++v)
            verts.push_back({buf[3+v*3], buf[4+v*3], buf[5+v*3], nx, ny, nz});
    }

    GL::Buffer vbo;
    vbo.setData(Containers::ArrayView<const void>(verts.data(),
                verts.size() * sizeof(Vertex)));

    GL::Mesh mesh;
    mesh.setPrimitive(GL::MeshPrimitive::Triangles)
        .setCount(static_cast<Int>(verts.size()))
        .addVertexBuffer(std::move(vbo), 0,
            Shaders::Phong::Position{},
            Shaders::Phong::Normal{});

    return mesh;
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

        Quaternion q{{p.qx, p.qy, p.qz}, p.qw};
        Matrix4 body_tf = Matrix4::from(q.normalized().toMatrix(), {p.x, p.y, p.z});

        Vector3 scale{sp.half_x, sp.half_y, sp.half_z};
        if (scale == Vector3{}) scale = {0.1f, 0.1f, 0.1f};

        Matrix4 model = body_tf * Matrix4::scaling(scale);
        Matrix4 mv    = _view * model;

        _shader.setDiffuseColor(bodyColor(i, scene))
               .setTransformationMatrix(mv)
               .setNormalMatrix(mv.normalMatrix())
               .setProjectionMatrix(_proj);

        // If body has a meshmap entry, use its STL mesh (unscaled — STL is real-size).
        auto it = _meshmap.find(i);
        if (it != _meshmap.end()) {
            const std::string& stl_path = it->second;
            auto cache_it = _mesh_cache.find(stl_path);
            if (cache_it == _mesh_cache.end()) {
                _mesh_cache.emplace(stl_path, load_stl_mesh(stl_path));
                cache_it = _mesh_cache.find(stl_path);
            }
            // STL meshes are in their own real-world units; use body pose only (no shape scale).
            Matrix4 mv_stl = _view * body_tf;
            _shader.setTransformationMatrix(mv_stl)
                   .setNormalMatrix(mv_stl.normalMatrix());
            _shader.draw(cache_it->second);
        } else {
            bool is_sphere = (sp.type == ShapeType::Sphere);
            _shader.draw(is_sphere ? _sphere : _box);
        }
    }
}

Color3 Renderer::bodyColor(uint32_t i, const SceneFileDesc& scene) {
    static const Color3 palette[] = {
        0x4466cc_rgbf, 0x44cc66_rgbf, 0xcc6644_rgbf, 0xccaa44_rgbf,
        0x44cccc_rgbf, 0xcc44cc_rgbf, 0xaacc44_rgbf, 0x4488ff_rgbf,
    };
    (void)scene;
    return palette[i % 8];
}

} // namespace dyphur::viz
