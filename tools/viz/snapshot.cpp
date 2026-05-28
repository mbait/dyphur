#include "snapshot.hpp"
#include "renderer.hpp"
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Image.h>
#include <Magnum/ImageView.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/PixelFormat.h>
#include <Magnum/Platform/WindowlessEglApplication.h>
#include <Magnum/Trade/AbstractImageConverter.h>
#include <Corrade/PluginManager/Manager.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace dyphur::viz {

using namespace Magnum;
using namespace Magnum::Math::Literals;

class SnapshotApp : public Platform::WindowlessEglApplication {
public:
    explicit SnapshotApp(const Arguments& args,
                         std::string prefix, int frame_idx,
                         std::string outfile, int width, int height)
        : Platform::WindowlessEglApplication{args}
        , _prefix{std::move(prefix)}
        , _frame_idx{frame_idx}
        , _outfile{std::move(outfile)}
        , _width{width}
        , _height{height}
    {}

    int result() const { return _result; }

    int exec() override {
        _result = render();
        return _result;
    }

    int render() {
        SceneFileDesc scene;
        try {
            scene = read_scene(_prefix);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "snapshot: %s\n", e.what());
            return 1;
        }

        std::ifstream traj(_prefix + ".trajectory", std::ios::binary);
        if (!traj) {
            std::fprintf(stderr, "snapshot: cannot open %s.trajectory\n", _prefix.c_str());
            return 1;
        }
        uint32_t hdr[2];
        traj.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        const uint32_t n_bodies = hdr[0];

        const std::streamoff frame_bytes =
            static_cast<std::streamoff>(n_bodies) * 7 * sizeof(float);
        traj.seekg(static_cast<std::streamoff>(sizeof(hdr)) +
                   static_cast<std::streamoff>(_frame_idx) * frame_bytes);
        if (!traj) {
            std::fprintf(stderr, "snapshot: frame %d not found\n", _frame_idx);
            return 1;
        }

        std::vector<BodyPose> poses(n_bodies);
        for (uint32_t i = 0; i < n_bodies; ++i) {
            float row[7];
            traj.read(reinterpret_cast<char*>(row), sizeof(row));
            poses[i] = {row[0], row[1], row[2], row[3], row[4], row[5], row[6]};
        }

        // Offscreen framebuffer.
        GL::Renderbuffer color_rb, depth_rb;
        color_rb.setStorage(GL::RenderbufferFormat::RGBA8, {_width, _height});
        depth_rb.setStorage(GL::RenderbufferFormat::Depth24Stencil8, {_width, _height});

        GL::Framebuffer fb{Range2Di{{}, {_width, _height}}};
        fb.attachRenderbuffer(GL::Framebuffer::ColorAttachment{0}, color_rb);
        fb.attachRenderbuffer(GL::Framebuffer::BufferAttachment::DepthStencil, depth_rb);
        fb.bind();

        GL::Renderer::setClearColor(0x222222_rgbf);
        fb.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);

        const float aspect = float(_width) / float(_height);
        Matrix4 proj = Matrix4::perspectiveProjection(55.0_degf, aspect, 0.1f, 200.f);
        Matrix4 view = Matrix4::lookAt({8.f, 8.f, 12.f}, {0.f, 2.f, 0.f}, {0.f, 1.f, 0.f});

        Renderer renderer;
        renderer.setViewProjection(view, proj);
        renderer.draw(scene, poses);

        Image2D image{PixelFormat::RGBA8Unorm};
        fb.read(fb.viewport(), image);

        Corrade::PluginManager::Manager<Trade::AbstractImageConverter> manager;
        auto converter = manager.loadAndInstantiate("StbImageConverter");
        if (!converter) {
            std::fprintf(stderr, "snapshot: cannot load StbImageConverter\n");
            return 1;
        }
        if (!converter->exportToFile(image, _outfile)) {
            std::fprintf(stderr, "snapshot: cannot write %s\n", _outfile.c_str());
            return 1;
        }

        std::printf("snapshot: wrote %s (%dx%d, frame %d of %s)\n",
                    _outfile.c_str(), _width, _height, _frame_idx, _prefix.c_str());
        return 0;
    }

    std::string _prefix;
    int         _frame_idx;
    std::string _outfile;
    int         _width, _height;
    int         _result = 0;
};

int run_snapshot(int argc, char** argv) {
    std::string prefix;
    int frame_idx   = 0;
    std::string outfile = "snapshot.png";
    int width  = 1280;
    int height = 720;

    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--frame") == 0 && i + 1 < argc)
            frame_idx = std::atoi(argv[++i]);
        else if ((std::strcmp(argv[i], "-o") == 0 ||
                  std::strcmp(argv[i], "--output") == 0) && i + 1 < argc)
            outfile = argv[++i];
        else if (std::strcmp(argv[i], "--width") == 0 && i + 1 < argc)
            width = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "--height") == 0 && i + 1 < argc)
            height = std::atoi(argv[++i]);
        else if (argv[i][0] != '-' && prefix.empty() &&
                 std::strcmp(argv[i], "snapshot") != 0)
            prefix = argv[i];
    }

    if (prefix.empty()) {
        std::fprintf(stderr,
            "usage: viz snapshot <prefix> [--frame N] [-o out.png] "
            "[--width W] [--height H]\n");
        return 1;
    }

    SnapshotApp app{{argc, argv}, prefix, frame_idx, outfile, width, height};
    app.exec();
    return app.result();
}

} // namespace dyphur::viz
