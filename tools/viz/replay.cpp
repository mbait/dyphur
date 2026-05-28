#include "replay.hpp"
#include "renderer.hpp"
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/Math/Matrix4.h>
#include <Magnum/Platform/GlfwApplication.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace dyphur::viz {

using namespace Magnum;
using namespace Magnum::Math::Literals;

class ReplayApp : public Platform::GlfwApplication {
public:
    explicit ReplayApp(const Arguments& args,
                       std::string prefix, float fps, bool loop)
        : Platform::GlfwApplication{args,
            Configuration{}.setTitle("dyphur viz — " + prefix)
                           .setSize({1280, 720})}
        , _prefix{std::move(prefix)}
        , _frame_dt{1.f / fps}
        , _loop{loop}
    {
        load();
    }

private:
    void load() {
        try {
            _scene = read_scene(_prefix);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "replay: %s\n", e.what());
            exit(1);
        }

        std::ifstream traj(_prefix + ".trajectory", std::ios::binary);
        if (!traj) {
            std::fprintf(stderr, "replay: cannot open %s.trajectory\n", _prefix.c_str());
            exit(1);
        }
        uint32_t hdr[2];
        traj.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
        const uint32_t n_bodies = hdr[0];

        const long frame_bytes = static_cast<long>(n_bodies) * 7 * sizeof(float);
        std::vector<float> row(n_bodies * 7);
        while (traj.read(reinterpret_cast<char*>(row.data()), frame_bytes)) {
            std::vector<BodyPose> frame(n_bodies);
            for (uint32_t i = 0; i < n_bodies; ++i) {
                float* r = row.data() + i * 7;
                frame[i] = {r[0], r[1], r[2], r[3], r[4], r[5], r[6]};
            }
            _frames.push_back(std::move(frame));
        }

        if (_frames.empty()) {
            std::fprintf(stderr, "replay: no frames in %s.trajectory\n", _prefix.c_str());
            exit(1);
        }

        std::printf("replay: %zu bodies, %zu frames, %.1f fps playback\n",
                    _scene.n_bodies, _frames.size(), 1.f / _frame_dt);

        _renderer = std::make_unique<Renderer>();
        updateCamera();
    }

    void drawEvent() override {
        GL::defaultFramebuffer.clear(GL::FramebufferClear::Color | GL::FramebufferClear::Depth);
        GL::Renderer::setClearColor(0x222222_rgbf);

        if (!_paused) {
            _time_acc += 1.f / 60.f; // assume ~60 fps window refresh
            while (_time_acc >= _frame_dt) {
                _time_acc -= _frame_dt;
                if (_cur_frame + 1 < static_cast<int>(_frames.size()))
                    ++_cur_frame;
                else if (_loop)
                    _cur_frame = 0;
            }
        }

        _renderer->draw(_scene, _frames[_cur_frame]);
        swapBuffers();
        redraw();
    }

    void keyPressEvent(KeyEvent& e) override {
        if (e.key() == KeyEvent::Key::Space) {
            _paused = !_paused;
        } else if (e.key() == KeyEvent::Key::Right) {
            _paused = true;
            if (_cur_frame + 1 < static_cast<int>(_frames.size())) ++_cur_frame;
        } else if (e.key() == KeyEvent::Key::Left) {
            _paused = true;
            if (_cur_frame > 0) --_cur_frame;
        } else if (e.key() == KeyEvent::Key::Q || e.key() == KeyEvent::Key::Esc) {
            exit(0);
        }
        e.setAccepted();
        redraw();
    }

    void mousePressEvent(MouseEvent& e) override {
        if (e.button() == MouseEvent::Button::Left)
            _last_mouse = {e.position().x(), e.position().y()};
        e.setAccepted();
    }

    void mouseMoveEvent(MouseMoveEvent& e) override {
        if (!(e.buttons() & MouseMoveEvent::Button::Left)) return;
        auto pos = e.position();
        float dx = float(pos.x() - _last_mouse[0]);
        float dy = float(pos.y() - _last_mouse[1]);
        _last_mouse = {pos.x(), pos.y()};
        _azimuth   -= dx * 0.005f;
        _elevation += dy * 0.005f;
        _elevation  = Math::clamp(_elevation, -1.5f, 1.5f);
        updateCamera();
        e.setAccepted();
        redraw();
    }

    void mouseScrollEvent(MouseScrollEvent& e) override {
        _distance *= std::exp(-e.offset().y() * 0.1f);
        _distance  = Math::clamp(_distance, 1.f, 200.f);
        updateCamera();
        e.setAccepted();
        redraw();
    }

    void viewportEvent(ViewportEvent& e) override {
        GL::defaultFramebuffer.setViewport({{}, e.framebufferSize()});
        updateCamera();
        redraw();
    }

    void updateCamera() {
        if (!_renderer) return;
        float cx = _distance * std::cos(_elevation) * std::sin(_azimuth);
        float cy = _distance * std::sin(_elevation);
        float cz = _distance * std::cos(_elevation) * std::cos(_azimuth);
        Matrix4 view = Matrix4::lookAt({cx, cy, cz}, {0.f, 0.f, 0.f}, {0.f, 1.f, 0.f});

        auto sz = GL::defaultFramebuffer.viewport().size();
        float aspect = sz.x() > 0 && sz.y() > 0
            ? float(sz.x()) / float(sz.y()) : 16.f / 9.f;
        Matrix4 proj = Matrix4::perspectiveProjection(55.0_degf, aspect, 0.1f, 200.f);

        _renderer->setViewProjection(view, proj);
    }

    std::string              _prefix;
    float                    _frame_dt;
    bool                     _loop;
    SceneFileDesc            _scene;
    std::vector<std::vector<BodyPose>> _frames;
    std::unique_ptr<Renderer> _renderer;

    int   _cur_frame  = 0;
    float _time_acc   = 0.f;
    bool  _paused     = false;

    float _azimuth   = 0.5f;
    float _elevation = 0.5f;
    float _distance  = 15.f;
    int   _last_mouse[2] = {0, 0};
};

int run_replay(int argc, char** argv) {
    std::string prefix;
    float fps  = 30.f;
    bool  loop = false;

    for (int i = 0; i < argc; ++i) {
        if (std::strcmp(argv[i], "--fps") == 0 && i + 1 < argc)
            fps = std::atof(argv[++i]);
        else if (std::strcmp(argv[i], "--loop") == 0)
            loop = true;
        else if (argv[i][0] != '-' && prefix.empty() && std::strcmp(argv[i], "replay") != 0)
            prefix = argv[i];
    }

    if (prefix.empty()) {
        std::fprintf(stderr, "usage: viz replay <prefix> [--fps N] [--loop]\n");
        return 1;
    }

    ReplayApp app{{argc, argv}, prefix, fps, loop};
    return app.exec();
}

} // namespace dyphur::viz
