// v0.1 demo: headless rigid-body stress test — 1 025 bodies (1 024 boxes + ground).
//
// Writes three output files (prefix defaults to "stress_test_blocks"):
//   <prefix>.trajectory      binary per-frame state dump (30 Hz snapshots)
//   <prefix>.metrics.json    step/contact/timing statistics
//   <prefix>.golden          FNV-1a-64 hash of the final body state (for CI)
//
// Usage: stress_test_blocks [out_prefix] [n_frames] [--cpu]

#include <core/body_store.hpp>
#include <core/broadphase.hpp>
#include <core/integrator.hpp>
#include <core/narrowphase.hpp>
#include <core/shape_store.hpp>
#include <core/xpbd_solver.hpp>
#include <compute/device.hpp>
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace dyphur;
using Clock   = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

static uint64_t fnv1a64(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 0x00000100000001B3ULL;
    }
    return h;
}

int main(int argc, char** argv) {
    std::string prefix  = "stress_test_blocks";
    int         n_frames = 300;

    bool force_cpu = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu") { force_cpu = true; continue; }
        if (prefix == "stress_test_blocks") prefix = argv[i];
        else n_frames = std::atoi(argv[i]);
    }

    spdlog::info("dyphur stress_test_blocks: {} frames, prefix='{}'", n_frames, prefix);

    Device dev = [&]() -> Device {
        if (force_cpu) return Device::default_cpu();
        try { return Device::default_gpu(); }
        catch (...) { return Device::default_cpu(); }
    }();
    auto s   = dev.make_stream();
    auto& q  = s.queue();

    // ── Scene dimensions ──────────────────────────────────────────────────────
    constexpr int NX = 8, NZ = 8, NY = 16;          // 8×16×8 = 1 024 dynamic boxes
    constexpr int N_DYN    = NX * NY * NZ;
    constexpr int N_BODIES = N_DYN + 1;              // +1 static ground

    constexpr uint32_t MAX_PAIRS    = 65536u;
    constexpr uint32_t MAX_CONTACTS = 16384u;

    constexpr float BOX_H  = 0.4f;   // half-extent of dynamic boxes
    constexpr float GND_HY = 0.5f;   // ground half-height
    constexpr float GND_HX = 20.f;   // ground half-width/depth

    // ── Shapes ────────────────────────────────────────────────────────────────
    ShapeStore ss(s, 4);

    ShapeParams dyn_sp;
    dyn_sp.type  = ShapeType::Box;
    dyn_sp.half_x = BOX_H; dyn_sp.half_y = BOX_H; dyn_sp.half_z = BOX_H;
    uint32_t dyn_shape = ss.add(dyn_sp);

    ShapeParams gnd_sp;
    gnd_sp.type  = ShapeType::Box;
    gnd_sp.half_x = GND_HX; gnd_sp.half_y = GND_HY; gnd_sp.half_z = GND_HX;
    uint32_t gnd_shape = ss.add(gnd_sp);

    ss.upload();

    // ── Bodies ────────────────────────────────────────────────────────────────
    BodyStore bs(s, N_BODIES);

    // Box inertia: I = m/3 * (hy^2 + hz^2) for a solid box (here a cube).
    const float m      = 1.f;
    const float I_diag = m / 3.f * (BOX_H * BOX_H + BOX_H * BOX_H);   // 2mh²/3

    BodyParams dp;
    dp.mass     = m;
    dp.inertia  = Mat3f(I_diag, 0.f, 0.f,
                        0.f, I_diag, 0.f,
                        0.f, 0.f, I_diag);
    dp.shape_handle = dyn_shape;
    dp.flags    = 0;

    for (int iy = 0; iy < NY; ++iy)
        for (int ix = 0; ix < NX; ++ix)
            for (int iz = 0; iz < NZ; ++iz) {
                dp.position = {
                    (ix - NX / 2 + 0.5f) * 1.0f,
                    GND_HY + (iy + 1) * 1.0f,        // stacked above ground
                    (iz - NZ / 2 + 0.5f) * 1.0f
                };
                bs.add(dp);
            }

    // Static ground plate
    BodyParams gp;
    gp.mass         = 0.f;
    gp.position     = {0.f, 0.f, 0.f};
    gp.shape_handle = gnd_shape;
    gp.flags        = BodyFlag::Static;
    bs.add(gp);

    bs.upload();
    s.wait();

    spdlog::info("Device: {} ({})", dev.name(), dev.is_gpu() ? "GPU" : "CPU");
    spdlog::info("Bodies: {} dynamic + 1 static ground", N_DYN);

    // ── Physics objects ───────────────────────────────────────────────────────
    constexpr float DT = 1.f / 60.f;
    const AABB scene_bounds = { -25.f, -2.f, -25.f, 25.f, 22.f, 25.f };

    Broadphase  bp(s, N_BODIES, MAX_PAIRS);
    Narrowphase np(s, MAX_CONTACTS);
    XpbdSolver  solver(s, 10);

    IntegratorParams ip;
    ip.gravity    = { 0.f, -9.81f, 0.f };
    ip.dt         = DT;
    ip.n_substeps = 1;

    BodyView bv = bs.view();
    ShapeView sv = ss.view();

    // ── Trajectory file ───────────────────────────────────────────────────────
    std::ofstream traj(prefix + ".trajectory", std::ios::binary);
    {
        uint32_t hdr[2] = { static_cast<uint32_t>(N_BODIES),
                            static_cast<uint32_t>(n_frames) };
        traj.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    }

    std::vector<float> hx(N_BODIES), hy(N_BODIES), hz(N_BODIES);
    std::vector<float> hw(N_BODIES), hqx(N_BODIES), hqy(N_BODIES), hqz(N_BODIES);

    auto download_state = [&]() {
        q.memcpy(hx.data(),  bv.pos_x, N_BODIES * sizeof(float));
        q.memcpy(hy.data(),  bv.pos_y, N_BODIES * sizeof(float));
        q.memcpy(hz.data(),  bv.pos_z, N_BODIES * sizeof(float));
        q.memcpy(hw.data(),  bv.rot_w, N_BODIES * sizeof(float));
        q.memcpy(hqx.data(), bv.rot_x, N_BODIES * sizeof(float));
        q.memcpy(hqy.data(), bv.rot_y, N_BODIES * sizeof(float));
        q.memcpy(hqz.data(), bv.rot_z, N_BODIES * sizeof(float)).wait();
    };

    auto write_traj_frame = [&]() {
        for (int i = 0; i < N_BODIES; ++i) {
            float row[7] = { hx[i], hy[i], hz[i], hw[i], hqx[i], hqy[i], hqz[i] };
            traj.write(reinterpret_cast<const char*>(row), sizeof(row));
        }
    };

    // ── Main simulation loop ──────────────────────────────────────────────────
    uint64_t total_contacts = 0;
    auto wall_start = Clock::now();

    for (int frame = 0; frame < n_frames; ++frame) {
        integrate(s, bv, ip);
        s.wait();

        bp.build_and_query(s, bv, sv, scene_bounds);
        s.wait();
        uint32_t n_pairs = bp.download_count(s);
        bp.sort_pairs(s, n_pairs);
        s.wait();

        np.run(s, bp.pairs_ptr(), n_pairs, bv, sv);
        s.wait();
        uint32_t n_contacts = np.download_count(s);
        total_contacts += n_contacts;

        solver.solve(s, np.contacts(), bv, DT);
        s.wait();

        // Snapshot trajectory at 30 Hz (every other frame)
        if (frame % 2 == 0) {
            download_state();
            write_traj_frame();
        }
    }

    traj.close();

    double wall_sec = Seconds(Clock::now() - wall_start).count();
    double fps      = n_frames / wall_sec;
    double avg_cnt  = static_cast<double>(total_contacts) / n_frames;

    // ── Final-state hash (determinism gate) ──────────────────────────────────
    download_state();
    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    uint64_t hash = FNV_OFFSET;
    hash = fnv1a64(hash, hx.data(),  N_BODIES * sizeof(float));
    hash = fnv1a64(hash, hy.data(),  N_BODIES * sizeof(float));
    hash = fnv1a64(hash, hz.data(),  N_BODIES * sizeof(float));
    hash = fnv1a64(hash, hw.data(),  N_BODIES * sizeof(float));
    hash = fnv1a64(hash, hqx.data(), N_BODIES * sizeof(float));
    hash = fnv1a64(hash, hqy.data(), N_BODIES * sizeof(float));
    hash = fnv1a64(hash, hqz.data(), N_BODIES * sizeof(float));

    {
        std::ofstream f(prefix + ".golden");
        f << fmt::format("{:016x}\n", hash);
    }

    // ── Metrics JSON ──────────────────────────────────────────────────────────
    {
        std::ofstream f(prefix + ".metrics.json");
        f << fmt::format(
            "{{\n"
            "  \"n_bodies\": {},\n"
            "  \"n_frames\": {},\n"
            "  \"simulation_seconds\": {:.4f},\n"
            "  \"wall_clock_seconds\": {:.4f},\n"
            "  \"frames_per_second\": {:.2f},\n"
            "  \"avg_contacts_per_frame\": {:.1f},\n"
            "  \"realtime_factor\": {:.3f}\n"
            "}}\n",
            N_BODIES, n_frames,
            n_frames * static_cast<double>(DT),
            wall_sec, fps, avg_cnt,
            fps / 60.0
        );
    }

    spdlog::info("Completed {} frames in {:.3f}s  ({:.1f} fps, {:.2f}x realtime)",
                 n_frames, wall_sec, fps, fps / 60.0);
    spdlog::info("Avg contacts/frame: {:.1f}", avg_cnt);
    spdlog::info("Hash: {:016x}", hash);
    spdlog::info("Wrote: {0}.trajectory  {0}.metrics.json  {0}.golden", prefix);

    return 0;
}
