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
#include <core/articulation.hpp>
#include <compute/device.hpp>
#include "../common/scene_io.hpp"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <algorithm>
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

    bool force_cpu = false, profile = false;
    int  gnx = 8, gny = 16, gnz = 8;   // default grid = 1 024 dynamic boxes
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu")     { force_cpu = true; continue; }
        if (a == "--profile") { profile   = true; continue; }
        if (a == "--grid" && i + 3 < argc) {
            gnx = std::atoi(argv[++i]); gny = std::atoi(argv[++i]); gnz = std::atoi(argv[++i]);
            continue;
        }
        if (prefix == "stress_test_blocks") prefix = a;
        else n_frames = std::atoi(a.c_str());
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
    const int NX = gnx, NZ = gnz, NY = gny;         // default 8×16×8 = 1 024 boxes
    const int N_DYN    = NX * NY * NZ;
    const int N_BODIES = N_DYN + 1;                  // +1 static ground

    // Capacities scale with body count (the grid can now be 4k/8k+ via --grid).
    // Sized for the transient peak during settling, not steady state: a collapsing
    // 32-high stack briefly hits ~8 contacts/body and ~20 candidate pairs/body, well
    // above the settled ~3–5. Under-sizing silently drops pairs/contacts in atomic
    // order → non-deterministic results (was Issue 6); overflow is detected below.
    const uint32_t MAX_PAIRS    = std::max<uint32_t>(65536u, static_cast<uint32_t>(N_DYN) * 48u);
    const uint32_t MAX_CONTACTS = std::max<uint32_t>(16384u, static_cast<uint32_t>(N_DYN) * 16u);

    constexpr float BOX_H  = 0.4f;   // half-extent of dynamic boxes
    constexpr float GND_HY = 0.5f;   // ground half-height
    // Ground half-width grows with the footprint so all stacks land on the plate.
    const float GND_HX = std::max(20.f, 0.5f * static_cast<float>(std::max(NX, NZ)) + 2.f);

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

    // ── Scene descriptor ──────────────────────────────────────────────────────
    {
        std::vector<uint32_t> body_scene_idx(N_BODIES);
        std::fill(body_scene_idx.begin(), body_scene_idx.begin() + N_DYN, dyn_shape);
        body_scene_idx[N_DYN] = gnd_shape;
        ShapeParams scene_shapes[] = {dyn_sp, gnd_sp};
        write_scene(prefix, N_BODIES, body_scene_idx.data(), scene_shapes, 2);
    }

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
    const float bound_xz = GND_HX + 5.f;
    const float bound_y  = std::max(22.f, GND_HY + static_cast<float>(NY) + 5.f);
    const AABB scene_bounds = { -bound_xz, -2.f, -bound_xz, bound_xz, bound_y, bound_xz };

    Broadphase  bp(s, N_BODIES, MAX_PAIRS);
    Narrowphase np(s, MAX_CONTACTS);
    XpbdSolver  solver(s, 10, MAX_CONTACTS);   // lambda accumulator must fit all contacts

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

    // ── Profiling mode: per-stage timing with s.wait() barriers ──────────────
    // Each stage is fenced so the wall time attributed to it is the real GPU
    // cost (this adds sync overhead the pipelined run does not pay, so the
    // stage-summed fps is a lower bound). Skips trajectory/golden output.
    if (profile) {
        const int N_WARM = 30;   // discard JIT + transient settling
        spdlog::info("PROFILE: {} bodies, {} warmup + {} measured frames, grid {}x{}x{}",
                     N_DYN, N_WARM, n_frames, NX, NY, NZ);

        double t_int = 0, t_bp = 0, t_dl = 0, t_sort = 0, t_np = 0, t_solve = 0;
        int    measured = 0;
        uint32_t n_pairs = 0;

        for (int frame = 0; frame < N_WARM + n_frames; ++frame) {
            const bool meas = frame >= N_WARM;
            Clock::time_point a;

            a = Clock::now(); integrate(s, bv, ip); s.wait();
            if (meas) t_int += Seconds(Clock::now() - a).count();

            a = Clock::now(); bp.build_and_query(s, bv, sv, scene_bounds); s.wait();
            if (meas) t_bp += Seconds(Clock::now() - a).count();

            a = Clock::now(); n_pairs = bp.download_count(s);
            if (meas) t_dl += Seconds(Clock::now() - a).count();

            a = Clock::now(); bp.sort_pairs(s, n_pairs); s.wait();
            if (meas) t_sort += Seconds(Clock::now() - a).count();

            a = Clock::now(); np.run(s, bp.pairs_ptr(), n_pairs, bv, sv); s.wait();
            if (meas) t_np += Seconds(Clock::now() - a).count();

            a = Clock::now(); solver.solve(s, np.contacts(), JointView{}, bv, DT); s.wait();
            if (meas) { t_solve += Seconds(Clock::now() - a).count(); ++measured; }
        }
        const uint32_t n_contacts = np.download_count(s);

        const double k = 1000.0 / measured;   // → ms / frame
        const double tot = (t_int + t_bp + t_dl + t_sort + t_np + t_solve) * k;
        auto row = [&](const char* name, double t) {
            const double ms = t * k;
            spdlog::info("  {:<16} {:8.3f} ms   {:5.1f}%", name, ms, 100.0 * ms / tot);
        };
        spdlog::info("── Per-stage avg over {} warm frames ({} bodies, ~{} contacts/frame, {} pairs) ──",
                     measured, N_DYN, n_contacts, n_pairs);
        row("integrate",        t_int);
        row("build_and_query",  t_bp);
        row("download_count",   t_dl);
        row("sort_pairs",       t_sort);
        row("narrowphase",      t_np);
        row("solve (XPBD)",     t_solve);
        spdlog::info("  {:<16} {:8.3f} ms   ({:.1f} fps, stage-summed lower bound)",
                     "TOTAL", tot, 1000.0 / tot);
        spdlog::info("Device: {} ({})", dev.name(), dev.is_gpu() ? "GPU" : "CPU");
        return 0;
    }

    // ── Main simulation loop ──────────────────────────────────────────────────
    auto wall_start = Clock::now();

    // Track buffer-capacity peaks. Both counters hold the *true* count (atomics keep
    // counting past the cap), so a peak ≥ capacity means entries were dropped in
    // non-deterministic order — fatal to reproducibility. n_pairs is free (already
    // downloaded); np.last_contact_count() is a cached host value (no extra sync).
    uint32_t peak_pairs = 0, peak_contacts = 0;
    for (int frame = 0; frame < n_frames; ++frame) {
        integrate(s, bv, ip);
        bp.build_and_query(s, bv, sv, scene_bounds);
        uint32_t n_pairs = bp.download_count(s);  // one sync per frame
        bp.sort_pairs(s, n_pairs);                 // no-op when n_pairs <= 1
        np.run(s, bp.pairs_ptr(), n_pairs, bv, sv);  // always resets contact store
        peak_pairs    = std::max(peak_pairs, n_pairs);
        peak_contacts = std::max(peak_contacts, np.last_contact_count());
        solver.solve(s, np.contacts(), JointView{}, bv, DT);

        // Trajectory snapshot every other frame (download_state provides GPU sync).
        if (frame % 2 == 0) {
            download_state();
            write_traj_frame();
        }
    }
    // Sample contact count from the last frame only — approximate, avoids a
    // blocking sync inside the hot loop.
    s.wait();
    double avg_cnt = static_cast<double>(np.download_count(s));

    // Determinism guard: a peak at/above capacity means entries were dropped in
    // atomic-append order, which makes the run non-reproducible (Issue 6).
    if (peak_pairs >= MAX_PAIRS || peak_contacts >= MAX_CONTACTS) {
        spdlog::error("CAPACITY OVERFLOW — results are NON-deterministic. "
                      "peak pairs {}/{}, peak contacts {}/{}. Increase MAX_PAIRS/MAX_CONTACTS.",
                      peak_pairs, MAX_PAIRS, peak_contacts, MAX_CONTACTS);
    } else {
        spdlog::info("Peak usage: pairs {}/{}, contacts {}/{} (headroom OK)",
                     peak_pairs, MAX_PAIRS, peak_contacts, MAX_CONTACTS);
    }

    traj.close();

    double wall_sec = Seconds(Clock::now() - wall_start).count();
    double fps      = n_frames / wall_sec;

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
