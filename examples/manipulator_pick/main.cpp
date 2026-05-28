// v0.2 demo: 7-DOF Franka-like manipulator pick-and-place.
//
// Scene: static base + 7 revolute-jointed links + block pre-grasped by the
// end-effector via a Fixed joint.  PD motors drive each revolute joint through
// a scripted 4-waypoint pick-and-place trajectory (home → lower → swing → home).
// No broadphase/narrowphase — joint constraints and PD control are the focus.
//
// Writes three output files (prefix defaults to "manipulator_pick"):
//   <prefix>.trajectory      binary per-frame state dump (30 Hz)
//   <prefix>.metrics.json    timing / performance statistics
//   <prefix>.golden          FNV-1a-64 hash of final body state (determinism gate)
//
// Usage: manipulator_pick [out_prefix] [n_frames] [--cpu]

#include <core/body_store.hpp>
#include <core/contact_store.hpp>
#include <core/integrator.hpp>
#include <core/joint_store.hpp>
#include <core/xpbd_solver.hpp>
#include <compute/device.hpp>
#include "../common/scene_io.hpp"
#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace dyphur;
using Clock   = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

static uint64_t fnv1a64(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x00000100000001B3ULL; }
    return h;
}

static float smoothstep(float t) {
    t = t < 0.f ? 0.f : (t > 1.f ? 1.f : t);
    return t * t * (3.f - 2.f * t);
}

// ── Arm constants ─────────────────────────────────────────────────────────────

static constexpr int   N_ARM_JOINTS = 7;
static constexpr int   N_BODIES     = 9;  // base(0) + links(1-7) + block(8)

// Link half-extents (body-local frame: link axis along Y).
static constexpr float LK_HY = 0.15f;   // half-length along the link
static constexpr float LK_HX = 0.04f;   // half-width
static constexpr float LK_HZ = 0.04f;

static constexpr float LINK_MASS = 1.f;

// Joint spacing: each joint is 2*LK_HY = 0.30 m apart along the arm.
static constexpr float SEP = 2.f * LK_HY;

// Joint axes — alternating to give Franka-like 3D workspace.
static constexpr float AXES[N_ARM_JOINTS][3] = {
    {0.f, 1.f, 0.f},   // J0: waist spin (Y)
    {1.f, 0.f, 0.f},   // J1: shoulder pitch (X)
    {0.f, 1.f, 0.f},   // J2: upper-arm roll (Y)
    {1.f, 0.f, 0.f},   // J3: elbow pitch (X)
    {0.f, 1.f, 0.f},   // J4: forearm roll (Y)
    {1.f, 0.f, 0.f},   // J5: wrist pitch (X)
    {0.f, 0.f, 1.f},   // J6: wrist roll (Z)
};

// Franka-like joint limits (radians).
static constexpr float JNT_LO[N_ARM_JOINTS] = {-2.8973f,-1.7628f,-2.8973f,-3.0718f,-2.8973f,-0.0175f,-2.8973f};
static constexpr float JNT_HI[N_ARM_JOINTS] = { 2.8973f, 1.7628f, 2.8973f,-0.0698f, 2.8973f, 3.7525f, 2.8973f};

// PD controller gains.
static constexpr float STIFFNESS = 300.f;
static constexpr float DAMPING   =  40.f;

// ── Scripted trajectory ───────────────────────────────────────────────────────
//
// Four waypoints in joint space.  The arm lifts the block (already grasped),
// swings to the "place" location, then returns home.
//
static constexpr int   N_WP = 4;
static constexpr float WAYPOINTS[N_WP][N_ARM_JOINTS] = {
    { 0.f,   0.f,  0.f,   0.f,  0.f,  0.f,  0.f  },   // 0: home (block held aloft)
    { 0.f,  -0.8f, 0.f,  -1.8f, 0.f,  1.0f, 0.f  },   // 1: lower arm (reach toward place)
    { 1.0f, -0.8f, 0.f,  -1.8f, 0.f,  1.0f, 0.f  },   // 2: swing EE to side (place location)
    { 0.f,   0.f,  0.f,   0.f,  0.f,  0.f,  0.f  },   // 3: return home
};

// Frames (at 60 fps) to spend on each transition.
static constexpr int WP_FRAMES[N_WP] = {90, 150, 150, 210};

// Interpolate targets at a given frame.
static void trajectory_targets(int frame, float out[N_ARM_JOINTS]) {
    int phase = 0;
    int f = frame;
    while (phase < N_WP - 1 && f >= WP_FRAMES[phase]) {
        f -= WP_FRAMES[phase];
        ++phase;
    }
    int   dur   = WP_FRAMES[phase];
    float alpha = smoothstep(static_cast<float>(f) / static_cast<float>(dur));
    int   next  = (phase + 1 < N_WP) ? phase + 1 : phase;
    for (int j = 0; j < N_ARM_JOINTS; ++j)
        out[j] = WAYPOINTS[phase][j] * (1.f - alpha) + WAYPOINTS[next][j] * alpha;
}

// ── Box inertia helper ────────────────────────────────────────────────────────

static Mat3f box_inertia(float m, float hx, float hy, float hz) {
    // I = m/3 * diag(hy²+hz², hx²+hz², hx²+hy²)
    return Mat3f(m/3.f*(hy*hy+hz*hz), 0.f, 0.f,
                 0.f, m/3.f*(hx*hx+hz*hz), 0.f,
                 0.f, 0.f, m/3.f*(hx*hx+hy*hy));
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    std::string prefix   = "manipulator_pick";
    int         n_frames = 600;
    bool        force_cpu = false;

    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu") { force_cpu = true; continue; }
        if (prefix == "manipulator_pick") prefix = argv[i];
        else n_frames = std::atoi(argv[i]);
    }

    spdlog::info("dyphur manipulator_pick: {} frames, prefix='{}'", n_frames, prefix);

    Device dev = [&]() -> Device {
        if (force_cpu) return Device::default_cpu();
        try { return Device::default_gpu(); }
        catch (...) { return Device::default_cpu(); }
    }();
    auto s  = dev.make_stream();
    auto& q = s.queue();

    // ── Bodies ────────────────────────────────────────────────────────────────
    //
    // Home configuration: arm vertical, all joints = 0.
    //   Body 0 (base):  center at (0, 0, 0), static.
    //   Body i (link i, i=1..7): center at (0, i*SEP, 0).
    //   Body 8 (block): center at (0, 7*SEP + LK_HY + 0.05, 0).

    BodyStore bs(s, N_BODIES);

    // Base — static anchor.
    {
        BodyParams p;
        p.position = {0.f, 0.f, 0.f};
        p.mass     = 0.f;
        p.flags    = BodyFlag::Static;
        p.inertia  = Mat3f::identity();
        bs.add(p);
    }

    // Arm links (bodies 1-7).
    const Mat3f link_I = box_inertia(LINK_MASS, LK_HX, LK_HY, LK_HZ);
    for (int i = 1; i <= N_ARM_JOINTS; ++i) {
        BodyParams p;
        p.position = {0.f, static_cast<float>(i) * SEP, 0.f};
        p.mass     = LINK_MASS;
        p.inertia  = link_I;
        p.flags    = 0;
        bs.add(p);
    }

    // Block (body 8) — grasped by end-effector from the start.
    constexpr float BLOCK_HALF = 0.05f;
    const float block_y = static_cast<float>(N_ARM_JOINTS) * SEP + LK_HY + BLOCK_HALF;
    {
        BodyParams p;
        p.position = {0.f, block_y, 0.f};
        p.mass     = 0.5f;
        p.inertia  = box_inertia(0.5f, BLOCK_HALF, BLOCK_HALF, BLOCK_HALF);
        p.flags    = 0;
        bs.add(p);
    }

    bs.upload();
    s.wait();

    // ── Scene descriptor ──────────────────────────────────────────────────────
    {
        ShapeParams shapes[2];
        shapes[0].type   = ShapeType::Box;
        shapes[0].half_x = LK_HX; shapes[0].half_y = LK_HY; shapes[0].half_z = LK_HZ;
        shapes[1].type   = ShapeType::Box;
        shapes[1].half_x = BLOCK_HALF; shapes[1].half_y = BLOCK_HALF; shapes[1].half_z = BLOCK_HALF;
        uint32_t idx[N_BODIES];
        for (int i = 0; i < N_BODIES - 1; ++i) idx[i] = 0;
        idx[N_BODIES - 1] = 1;
        write_scene(prefix, N_BODIES, idx, shapes, 2);
    }

    // ── Joints ────────────────────────────────────────────────────────────────
    //
    // 7 revolute joints (arm) + 1 fixed joint (grasp).

    JointStore js(s, N_ARM_JOINTS + 1);

    for (int i = 0; i < N_ARM_JOINTS; ++i) {
        JointParams p;
        p.body_parent    = static_cast<uint32_t>(i);
        p.body_child     = static_cast<uint32_t>(i + 1);
        // Joint i sits between body i's top and body (i+1)'s bottom.
        p.anchor_parent  = {0.f, LK_HY, 0.f};    // top of parent link
        p.anchor_child   = {0.f, -LK_HY, 0.f};   // bottom of child link
        p.axis           = {AXES[i][0], AXES[i][1], AXES[i][2]};
        p.limit_lo       = JNT_LO[i];
        p.limit_hi       = JNT_HI[i];
        p.target_pos     = WAYPOINTS[0][i];
        p.target_vel     = 0.f;
        p.stiffness      = STIFFNESS;
        p.damping        = DAMPING;
        p.compliance_pos = 0.f;
        p.compliance_ang = 0.f;
        p.type           = JointType::Revolute;
        js.add(p);
    }

    // Grasp joint: Fixed, end-effector → block.
    {
        JointParams p;
        p.body_parent    = N_ARM_JOINTS;      // body 7 (EE link)
        p.body_child     = N_ARM_JOINTS + 1;  // body 8 (block)
        p.anchor_parent  = {0.f, LK_HY, 0.f};         // top of EE link
        p.anchor_child   = {0.f, -BLOCK_HALF, 0.f};   // bottom of block
        p.compliance_pos = 0.f;
        p.compliance_ang = 0.f;
        p.type           = JointType::Fixed;
        js.add(p);
    }

    js.upload();
    s.wait();

    // Empty contact store (no collision detection in this demo).
    ContactStore cs(s, 1);
    cs.reset(s);
    s.wait();

    // ── Solver ────────────────────────────────────────────────────────────────

    XpbdSolver solver(s, 20);

    IntegratorParams ip;
    ip.gravity    = {0.f, -9.81f, 0.f};
    ip.dt         = 1.f / 60.f;
    ip.n_substeps = 1;

    BodyView  bv = bs.view();
    JointView jv = js.view();

    spdlog::info("Device: {} ({})", dev.name(), dev.is_gpu() ? "GPU" : "CPU");
    spdlog::info("Bodies: {} (base + 7 links + block)", N_BODIES);
    spdlog::info("Joints: {} revolute + 1 fixed grasp", N_ARM_JOINTS);

    // ── Trajectory file ───────────────────────────────────────────────────────

    std::ofstream traj(prefix + ".trajectory", std::ios::binary);
    {
        uint32_t hdr[2] = {static_cast<uint32_t>(N_BODIES),
                           static_cast<uint32_t>(n_frames)};
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
            float row[7] = {hx[i], hy[i], hz[i], hw[i], hqx[i], hqy[i], hqz[i]};
            traj.write(reinterpret_cast<const char*>(row), sizeof(row));
        }
    };

    // ── Simulation loop ───────────────────────────────────────────────────────

    float targets[N_ARM_JOINTS];
    float zeros[N_ARM_JOINTS] = {};
    auto wall_start = Clock::now();

    for (int frame = 0; frame < n_frames; ++frame) {
        // Update PD targets for this frame.
        trajectory_targets(frame, targets);
        js.set_targets(targets, zeros);

        integrate(s, bv, ip);
        solver.solve(s, cs.view(), jv, bv, ip.dt);

        if (frame % 2 == 0) {
            download_state();
            write_traj_frame();
        }
    }
    s.wait();  // single end-of-batch sync for timing

    traj.close();

    double wall_sec = Seconds(Clock::now() - wall_start).count();
    double fps      = n_frames / wall_sec;

    // ── Final-state hash ──────────────────────────────────────────────────────

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
            "  \"n_joints\": {},\n"
            "  \"n_frames\": {},\n"
            "  \"simulation_seconds\": {:.4f},\n"
            "  \"wall_clock_seconds\": {:.4f},\n"
            "  \"frames_per_second\": {:.2f},\n"
            "  \"realtime_factor\": {:.3f}\n"
            "}}\n",
            N_BODIES, N_ARM_JOINTS + 1, n_frames,
            n_frames * static_cast<double>(ip.dt),
            wall_sec, fps, fps / 60.0
        );
    }

    spdlog::info("Completed {} frames in {:.3f}s  ({:.1f} fps, {:.2f}x realtime)",
                 n_frames, wall_sec, fps, fps / 60.0);
    spdlog::info("Hash: {:016x}", hash);
    spdlog::info("Wrote: {0}.trajectory  {0}.metrics.json  {0}.golden", prefix);

    return 0;
}
