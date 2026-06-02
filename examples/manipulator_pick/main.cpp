// v0.3 demo: Franka Panda pick-and-place with contact-based friction grasp.
//
// Scenario (z-up, gravity along -z):
//   - Franka Panda arm (11 kinematic bodies: base + links 1-7 + hand + 2 fingers)
//     follows a pre-programmed joint trajectory.
//   - Pickup cube (dynamic, 50 mm) starts at position A = (0.4, 0, 0.025).
//   - Tower of 4 cubes (dynamic, same size) at position B = (0, 0.4, 0.025..0.175).
//   - Gripper fingers are kinematic; friction from XPBD contact holds the cube.
//   - Fingers open → cube free-falls ~0.15 m onto tower.
//
// Kinematic FK computed each frame via Panda DH parameters from the URDF.
// All Panda links use box collision approximations; meshmap written for viz rendering.
//
// Writes:
//   <prefix>.trajectory   binary per-frame state dump (30 Hz)
//   <prefix>.scene        shape catalog for viz tools
//   <prefix>.meshmap      per-body STL mesh paths for viz rendering
//   <prefix>.metrics.json timing / performance statistics
//   <prefix>.golden       FNV-1a-64 hash of final body state (determinism gate)
//
// Usage: manipulator_pick [out_prefix] [n_frames] [--cpu] [--debug-fk]

#include <core/body_store.hpp>
#include <core/broadphase.hpp>
#include <core/contact_store.hpp>
#include <core/convex_hull_store.hpp>
#include <core/integrator.hpp>
#include <core/joint_store.hpp>
#include <core/mesh_bvh.hpp>
#include <core/narrowphase.hpp>
#include <core/shape_store.hpp>
#include <core/xpbd_solver.hpp>
#include <compute/device.hpp>
#include "../common/scene_io.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <spdlog/spdlog.h>
#include <fmt/format.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
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
    t = t < 0.f ? 0.f : t > 1.f ? 1.f : t;
    return t * t * (3.f - 2.f * t);
}

// ── Panda forward kinematics ──────────────────────────────────────────────────
//
// All revolute joints rotate about their local z-axis after the joint offset.
// Each entry: (translation xyz, roll about x) — pitch and yaw offsets are 0 for
// all Panda joints.

struct JointDH { float dx, dy, dz, roll; };

static constexpr JointDH PANDA_DH[7] = {
    { 0.f,      0.f,     0.333f,  0.f            },  // joint 1
    { 0.f,      0.f,     0.f,    -M_PI / 2.f     },  // joint 2
    { 0.f,     -0.316f,  0.f,     M_PI / 2.f     },  // joint 3
    { 0.0825f,  0.f,     0.f,     M_PI / 2.f     },  // joint 4
    {-0.0825f,  0.384f,  0.f,    -M_PI / 2.f     },  // joint 5
    { 0.f,      0.f,     0.f,     M_PI / 2.f     },  // joint 6
    { 0.088f,   0.f,     0.f,     M_PI / 2.f     },  // joint 7
};

// Returns world-space pose of each Panda body as (position, quaternion w/x/y/z).
// Output: [link0, link1..link7, hand, leftfinger, rightfinger] = 11 entries.
//   q[7]    — revolute joint angles (radians)
//   q_l     — left finger prismatic displacement (+y direction, m)
//   q_r     — right finger prismatic displacement (+y direction, m)
static std::vector<std::pair<Vec3f, Quatf>>
panda_fk(const float q[7], float q_l, float q_r)
{
    using Iso = Eigen::Isometry3f;
    using AA  = Eigen::AngleAxisf;
    using V3  = Eigen::Vector3f;
    using Qf  = Eigen::Quaternionf;

    auto make_pair = [](const Iso& T) -> std::pair<Vec3f, Quatf> {
        Qf eq(T.rotation()); eq.normalize();
        const auto& p = T.translation();
        return { {p.x(), p.y(), p.z()},
                 Quatf{eq.w(), Vec3f{eq.x(), eq.y(), eq.z()}} };
    };

    std::vector<std::pair<Vec3f, Quatf>> poses(11);

    Iso T = Iso::Identity();
    poses[0] = make_pair(T);  // link0 at world origin

    for (int i = 0; i < 7; ++i) {
        const auto& dh = PANDA_DH[i];
        T = T
            * Iso(Eigen::Translation3f(dh.dx, dh.dy, dh.dz))
            * Iso(AA(dh.roll, V3::UnitX()))
            * Iso(AA(q[i],   V3::UnitZ()));
        poses[i + 1] = make_pair(T);  // link1..link7
    }

    // link8 frame (fixed flange offset)
    T = T * Iso(Eigen::Translation3f(0.f, 0.f, 0.107f));

    // hand (fixed -π/4 rotation about z)
    T = T * Iso(AA(-M_PI / 4.f, V3::UnitZ()));
    poses[8] = make_pair(T);  // hand

    // fingers: base at +z 0.0584 from hand; left along +y, right along -y
    Iso T_fb = T * Iso(Eigen::Translation3f(0.f, 0.f, 0.0584f));
    Iso T_lf = T_fb * Iso(Eigen::Translation3f(0.f,  q_l, 0.f));
    Iso T_rf = T_fb * Iso(Eigen::Translation3f(0.f, -q_r, 0.f));

    // right finger in URDF has 180° yaw offset
    Iso T_rf_body = T_rf * Iso(AA(M_PI, V3::UnitZ()));
    poses[9]  = make_pair(T_lf);       // left finger
    poses[10] = make_pair(T_rf_body);  // right finger (mirrored)

    return poses;
}

// Grasp point = midpoint of the two finger centres (world space) for joints q.
static Eigen::Vector3f grasp_point(const float q[7], float q_finger) {
    auto p = panda_fk(q, q_finger, q_finger);
    Eigen::Vector3f lf(p[9].first.x,  p[9].first.y,  p[9].first.z);
    Eigen::Vector3f rf(p[10].first.x, p[10].first.y, p[10].first.z);
    return 0.5f * (lf + rf);
}

// ── Inverse kinematics ──────────────────────────────────────────────────────────
//
// Damped least-squares (Levenberg-Marquardt) IK: solve for q[7] so the grasp point
// reaches `target`.  Finite-difference 3×7 Jacobian; seeded with q_seed for
// continuity and to select the right elbow/base branch.
static void panda_ik(const float q_seed[7], const Eigen::Vector3f& target,
                     float q_finger, float q_out[7]) {
    Eigen::Matrix<float, 7, 1> q;
    for (int i = 0; i < 7; ++i) q[i] = q_seed[i];

    const float eps = 1e-4f, lambda = 0.08f;
    for (int iter = 0; iter < 400; ++iter) {
        float qa[7]; for (int i = 0; i < 7; ++i) qa[i] = q[i];
        Eigen::Vector3f cur = grasp_point(qa, q_finger);
        Eigen::Vector3f err = target - cur;
        if (err.norm() < 1e-4f) break;

        Eigen::Matrix<float, 3, 7> J;
        for (int j = 0; j < 7; ++j) {
            float qj[7]; for (int i = 0; i < 7; ++i) qj[i] = q[i];
            qj[j] += eps;
            Eigen::Vector3f pj = grasp_point(qj, q_finger);
            J.col(j) = (pj - cur) / eps;
        }
        Eigen::Matrix3f JJt = J * J.transpose()
                            + lambda * lambda * Eigen::Matrix3f::Identity();
        Eigen::Matrix<float, 7, 1> dq = J.transpose() * JJt.ldlt().solve(err);
        q += dq;
    }
    for (int i = 0; i < 7; ++i) q_out[i] = q[i];
}

// ── Box inertia ───────────────────────────────────────────────────────────────

static Mat3f box_inertia(float m, float hx, float hy, float hz) {
    return Mat3f(m/3.f*(hy*hy+hz*hz), 0.f,            0.f,
                 0.f,            m/3.f*(hx*hx+hz*hz), 0.f,
                 0.f,            0.f,            m/3.f*(hx*hx+hy*hy));
}

// ── Body shape parameters for each Panda link (box approximations) ────────────

// Half-extents (metres) for collision boxes of each Panda body.
// Ordered: link0, link1..link7, hand, finger (shared by both).
// Link0..hand use tiny shapes (0.001m) so they don't spuriously collide with
// the environment — only the finger bodies need real collision geometry for grasping.
// The finger pad (index 9) is sized to straddle the 50 mm cube robustly in x and z;
// FHY (0.010) is its half-extent in the squeeze (y) direction.
// Finger pad (index 9) sized to fully enclose the 50 mm cube in x so the box-box
// contact normal stays in the squeeze (y) axis and can't flip and eject the cube.
// hx=0.040 (wider than cube) gives x-drift margin; hz=0.016 grips the cube's
// middle band only — clear of its bottom edge and the table top, so the close
// doesn't catch the lower edge and pop the cube upward.  Substepping keeps the
// contact alive through the lift, so a tall pad is no longer needed.
static constexpr float PANDA_HX[10] = {0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.040f};
static constexpr float PANDA_HY[10] = {0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.010f};
static constexpr float PANDA_HZ[10] = {0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.001f,0.016f};

// ── Scene constants ───────────────────────────────────────────────────────────

static constexpr float CUBE_HALF  = 0.025f;   // 50 mm picked cube
static constexpr float CUBE_MASS  = 0.1f;
// Tower built from wider 80 mm cubes so the 50 mm picked cube has landing margin
// on the tower top (a same-width top makes the placed cube topple off).
static constexpr float TOWER_CUBE_HALF = 0.040f;
static constexpr float TOWER_CUBE_MASS = 0.3f;
static constexpr int   TOWER_N    = 2;         // cubes in the tower

// Position A: pickup cube — placed exactly under the grasp finger x.
// FK at grasp config [0,0.3,0,-2.2,0,2.5,0.785] → lf x≈0.556, z≈0.187.
// Cube placed 3 mm into table top for initial contact detection.
static constexpr float A_X = 0.556f, A_Y = 0.f;
static constexpr float TABLE_A_HZ   = 0.081f;         // table half-height → top at 0.162 m
static constexpr float TABLE_A_TOP  = 2.f * TABLE_A_HZ;
static constexpr float CUBE_A_Z     = TABLE_A_TOP + CUBE_HALF - 0.003f;  // 0.184 m

// Position B: tower.  Kept at a modest azimuth from A (≈45°) so the transport is
// mostly translation with limited gripper rotation — single-point box contacts
// can only transmit limited torque, so a big base swing slips the cube.
static constexpr float B_X = 0.38f, B_Y = 0.38f;

// Finger collision-pad half-extent in the squeeze (y) direction.
static constexpr float FHY = 0.010f;

// Finger open/close targets (prismatic displacement from hand centre, metres).
// Finger inner face at ∓(q - FHY); cube face at ∓CUBE_HALF (0.025).
//   open   q=0.060 → inner face ∓0.050 → 25 mm clearance, so the open fingers fully
//          clear the cube in y and the gripper can retract straight up without
//          raking the just-placed cube off the tower.
//   closed q=0.022 → inner face ∓0.012 → ~13 mm penetration into cube
// (The open width exceeds the real Panda's 40 mm limit; harmless for the kinematic
// gripper and needed because a real 8 cm-stroke gripper barely clears a 5 cm cube.)
static constexpr float Q_FINGER_OPEN   = 0.060f;
static constexpr float Q_FINGER_CLOSED = 0.022f;

// ── Joint trajectory (world-space waypoints solved via IK at startup) ──────────
//
// Eight phases.  Each has a world-space grasp-point target (reached via IK) and a
// finger opening, both at the phase END.  The descent (P1→P2) and lift (P3→P4) are
// vertical (x,y constant) so the gripper does not sweep the cube sideways.
//
// seg0 settle+approach — tower settles while gripper moves above cube (open)
// seg1 descend         — gripper down to cube centre (open)
// seg2 close           — fingers close around cube (gripper stationary)
// seg3 lift            — gripper straight up (closed)
// seg4 transport       — swing to above tower (closed)
// seg5 lower           — descend to just above tower (closed)
// seg6 release         — fingers open at place height → cube free-falls onto tower
// seg7 retract         — empty gripper lifts away
// seg8 settle          — hold while the cube settles

static constexpr int N_PHASES = 9;
//                                          settle desc close lift xport lower rel retr hold
static constexpr int PHASE_FRAMES[N_PHASES] = {110, 80, 60, 110, 200, 90, 70, 50, 30};
// Total: 800 frames = 13.3 s at 60 Hz.  Lift and transport are slow so friction
// can accelerate / swing the cube without the gripper outrunning it.

// Grasp / carry / place heights for the grasp point (finger midpoint).
static constexpr float GRASP_Z = TABLE_A_TOP + CUBE_HALF;        // 0.187 m (settled cube centre)
static constexpr float CARRY_Z = 0.30f;                          // lifted / transport height
// Tower top surface ≈ TOWER_N*2*CUBE_HALF.  Release a small gap above it so the
// cube has a short free-fall, as required by the scenario.
static constexpr float TOWER_TOP = TOWER_N * 2.f * TOWER_CUBE_HALF;  // top surface of the tower
// Lowest grasp-point that still keeps the finger pad bottom clear of the tower top
// (pad bottom = PLACE_Z - finger hz).  Minimises the free-fall so the cube lands
// flat instead of toppling off the narrow tower top.
static constexpr float PLACE_Z   = TOWER_TOP + 0.016f + 0.004f;  // pad clears tower by 4 mm

// Home joint configuration (phase 0; arm retracted, gripper up high).
static constexpr float Q_HOME[7] = {0.f, -0.785f, 0.f, -2.356f, 0.f, 1.571f, 0.785f};

// World-space grasp-point target at the end of each segment (config[0] = home).
// Segment i interpolates config[i]→config[i+1]; the last is a hold.
static const Eigen::Vector3f WP_POS[N_PHASES] = {
    {0.f,  0.f,  0.f    },   // 0 home (not IK)
    {A_X,  A_Y,  CARRY_Z},   // 1 pre-grasp above cube
    {A_X,  A_Y,  GRASP_Z},   // 2 at cube (fingers close next segment)
    {A_X,  A_Y,  GRASP_Z},   // 3 grasped
    {A_X,  A_Y,  CARRY_Z},   // 4 lifted
    {B_X,  B_Y,  CARRY_Z},   // 5 above tower (transported)
    {B_X,  B_Y,  PLACE_Z},   // 6 lowered to just above tower
    {B_X,  B_Y,  PLACE_Z},   // 7 released (gripper holds; cube free-falls)
    {B_X,  B_Y,  CARRY_Z},   // 8 retract empty gripper up
};

// Finger opening at the end of each segment.
static constexpr float Q_FINGER[N_PHASES] = {
    Q_FINGER_OPEN,    // 0 home
    Q_FINGER_OPEN,    // 1 approach
    Q_FINGER_OPEN,    // 2 at cube, still open
    Q_FINGER_CLOSED,  // 3 close around cube
    Q_FINGER_CLOSED,  // 4 lift
    Q_FINGER_CLOSED,  // 5 transport
    Q_FINGER_CLOSED,  // 6 lower onto tower (still gripping)
    Q_FINGER_OPEN,    // 7 release → cube free-falls the small gap onto tower
    Q_FINGER_OPEN,    // 8 retract (empty)
};

// Joint configs per phase — filled by compute_waypoints() at startup.
static float g_Q[N_PHASES][7];

// Solve IK for each phase's world-space target, seeding from the previous phase.
static void compute_waypoints() {
    for (int i = 0; i < 7; ++i) g_Q[0][i] = Q_HOME[i];
    float seed[7];
    for (int i = 0; i < 7; ++i) seed[i] = Q_HOME[i];
    for (int ph = 1; ph < N_PHASES; ++ph) {
        // Seed the first transport pose with the base joint already aimed at the
        // target azimuth, so IK finds a natural shoulder-swing solution close to
        // the lift pose (avoids a contorted elbow branch that would fling the cube).
        if (ph == 5) seed[0] = std::atan2(WP_POS[ph].y(), WP_POS[ph].x());
        panda_ik(seed, WP_POS[ph], Q_FINGER[ph], g_Q[ph]);
        for (int i = 0; i < 7; ++i) seed[i] = g_Q[ph][i];
    }
}

// Get joint angles and finger opening at a (possibly fractional) frame time.
static void get_targets_f(float frame_t, float q_out[7], float& q_f_out)
{
    int phase = 0;
    float f = frame_t;
    while (phase < N_PHASES - 1 && f >= float(PHASE_FRAMES[phase])) {
        f -= float(PHASE_FRAMES[phase]); ++phase;
    }
    int   next  = std::min(phase + 1, N_PHASES - 1);
    float alpha = smoothstep(f / static_cast<float>(PHASE_FRAMES[phase]));
    for (int j = 0; j < 7; ++j)
        q_out[j] = g_Q[phase][j] * (1.f - alpha) + g_Q[next][j] * alpha;
    q_f_out = Q_FINGER[phase] * (1.f - alpha) + Q_FINGER[next] * alpha;
}

// Integer-frame convenience wrapper.
static void get_targets(int frame, float q_out[7], float& q_f_out)
{
    get_targets_f(static_cast<float>(frame), q_out, q_f_out);
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv)
{
    std::string prefix   = "manipulator_pick";
    int         n_frames = 800;
    bool        force_cpu = false;
    bool        debug_fk  = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--cpu")      { force_cpu = true; continue; }
        if (a == "--debug-fk") { debug_fk  = true; continue; }
        if (prefix == "manipulator_pick") prefix = argv[i];
        else n_frames = std::atoi(argv[i]);
    }

    // Asset directory: next to the executable.
    std::filesystem::path exe(argv[0]);
    std::string asset_dir = (exe.parent_path() / "assets").string();

    spdlog::info("dyphur manipulator_pick: {} frames, prefix='{}'", n_frames, prefix);

    // Solve IK for all phase waypoints (fills g_Q).
    compute_waypoints();

    Device dev = [&]() -> Device {
        if (force_cpu) return Device::default_cpu();
        try { return Device::default_gpu(); }
        catch (...) { return Device::default_cpu(); }
    }();
    auto s = dev.make_stream();
    auto& q_queue = s.queue();

    // ── Debug FK ─────────────────────────────────────────────────────────────

    if (debug_fk) {
        // Verify IK: grasp point reached vs target for each phase.
        for (int ph = 0; ph < N_PHASES; ++ph) {
            Eigen::Vector3f gp = grasp_point(g_Q[ph], Q_FINGER[ph]);
            spdlog::info("config[{}] target=({:.3f},{:.3f},{:.3f}) reached=({:.3f},{:.3f},{:.3f}) "
                         "q=[{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}]",
                ph, WP_POS[ph].x(), WP_POS[ph].y(), WP_POS[ph].z(),
                gp.x(), gp.y(), gp.z(),
                g_Q[ph][0],g_Q[ph][1],g_Q[ph][2],g_Q[ph][3],g_Q[ph][4],g_Q[ph][5],g_Q[ph][6]);
        }
        return 0;
    }

    // ── Body layout ───────────────────────────────────────────────────────────
    //
    //  [0..10]   Panda bodies (link0 static, link1-hand-fingers kinematic)
    //  [11]      Ground plane (static)
    //  [12]      Table at A (static)
    //  [13]      Pickup cube (dynamic)
    //  [14..17]  Tower cubes 0-3 (dynamic, stacked)
    //
    static constexpr int N_PANDA  = 11;
    static constexpr int IDX_GND  = 11;
    static constexpr int IDX_TBL  = 12;
    static constexpr int IDX_PICK = 13;
    static constexpr int IDX_TWR  = 14;
    // N_BODIES = N_PANDA(11) + ground(1) + table(1) + pickup(1) + TOWER_N(2) = 16
    static constexpr int N_BODIES = N_PANDA + 1 + 1 + 1 + TOWER_N;

    // Compute initial FK pose at frame 0 for kinematic body placement.
    float q_init[7]; float qf_init;
    get_targets(0, q_init, qf_init);
    auto init_poses = panda_fk(q_init, qf_init, qf_init);

    // ── Shape store ───────────────────────────────────────────────────────────

    ShapeStore ss(s, N_BODIES);

    // Panda link shapes (link0..link7, hand, leftfinger, rightfinger)
    uint32_t panda_shape_handles[N_PANDA];
    for (int i = 0; i < 9; ++i) {  // link0..link7 + hand
        ShapeParams sp;
        sp.type   = ShapeType::Box;
        sp.half_x = PANDA_HX[i];
        sp.half_y = PANDA_HY[i];
        sp.half_z = PANDA_HZ[i];
        panda_shape_handles[i] = ss.add(sp);
    }
    // fingers (indices 9 and 10, same shape)
    {
        ShapeParams sp;
        sp.type   = ShapeType::Box;
        sp.half_x = PANDA_HX[9];
        sp.half_y = PANDA_HY[9];
        sp.half_z = PANDA_HZ[9];
        panda_shape_handles[9]  = ss.add(sp);
        panda_shape_handles[10] = ss.add(sp);
    }

    // Ground, table, pickup cube, tower cubes
    ShapeParams sp_gnd;  sp_gnd.type = ShapeType::Box;
    sp_gnd.half_x = 2.f; sp_gnd.half_y = 2.f; sp_gnd.half_z = 0.01f;
    uint32_t h_gnd = ss.add(sp_gnd);

    ShapeParams sp_tbl; sp_tbl.type = ShapeType::Box;
    sp_tbl.half_x = 0.10f; sp_tbl.half_y = 0.10f; sp_tbl.half_z = TABLE_A_HZ;
    uint32_t h_tbl = ss.add(sp_tbl);

    ShapeParams sp_cube; sp_cube.type = ShapeType::Box;
    sp_cube.half_x = CUBE_HALF;
    sp_cube.half_y = CUBE_HALF;
    sp_cube.half_z = CUBE_HALF;
    uint32_t h_cube = ss.add(sp_cube);

    ShapeParams sp_twr; sp_twr.type = ShapeType::Box;
    sp_twr.half_x = TOWER_CUBE_HALF;
    sp_twr.half_y = TOWER_CUBE_HALF;
    sp_twr.half_z = TOWER_CUBE_HALF;
    uint32_t h_tower[TOWER_N];
    for (int i = 0; i < TOWER_N; ++i) h_tower[i] = ss.add(sp_twr);

    ss.upload();
    ShapeView sv = ss.view();

    // ── Convex hull + mesh stores (empty — all shapes are Box primitives) ────

    ConvexHullStore hs(s, 1u, 4096u);
    hs.upload();
    ConvexHullView hv = hs.view();

    MeshBvhStore ms(s, 1u, 65536u, 65536u, 131072u);
    ms.upload();
    MeshBvhCatalogView mv = ms.view();

    // ── Body store ────────────────────────────────────────────────────────────

    BodyStore bs(s, N_BODIES);

    // Panda bodies
    for (int i = 0; i < N_PANDA; ++i) {
        BodyParams p;
        const auto& [pos, rot] = init_poses[i];
        p.position   = pos;
        p.rotation   = rot;
        p.shape_handle = panda_shape_handles[i];
        if (i == 0) {
            // link0: static base
            p.mass   = 0.f;
            p.flags  = BodyFlag::Static;
            p.inertia = Mat3f::identity();
        } else {
            // kinematic arm links: mass=0 so inv_mass=0 → solver treats as infinite mass.
            // BodyFlag::Kinematic tells the integrator to advance by the set velocity.
            p.mass    = 0.f;
            p.flags   = BodyFlag::Kinematic;
            p.inertia = Mat3f::identity();
        }
        bs.add(p);
    }

    // Ground plane
    {
        BodyParams p;
        p.position     = {0.f, 0.f, -0.01f};
        p.shape_handle = h_gnd;
        p.mass         = 0.f;
        p.flags        = BodyFlag::Static;
        p.inertia      = Mat3f::identity();
        bs.add(p);
    }

    // Table at A (static pedestal)
    {
        BodyParams p;
        p.position     = {A_X, A_Y, TABLE_A_HZ};
        p.shape_handle = h_tbl;
        p.mass         = 0.f;
        p.flags        = BodyFlag::Static;
        p.inertia      = Mat3f::identity();
        bs.add(p);
    }

    // Pickup cube at A (on top of table)
    {
        BodyParams p;
        p.position     = {A_X, A_Y, CUBE_A_Z};
        p.shape_handle = h_cube;
        p.mass         = CUBE_MASS;
        p.inertia      = box_inertia(CUBE_MASS, CUBE_HALF, CUBE_HALF, CUBE_HALF);
        bs.add(p);
    }

    // Tower cubes at B (each placed 3 mm into the surface below for contact detection).
    for (int i = 0; i < TOWER_N; ++i) {
        BodyParams p;
        p.position     = {B_X, B_Y, TOWER_CUBE_HALF - 0.003f + i * (2.f * TOWER_CUBE_HALF)};
        p.shape_handle = h_tower[i];
        p.mass         = TOWER_CUBE_MASS;
        p.inertia      = box_inertia(TOWER_CUBE_MASS, TOWER_CUBE_HALF, TOWER_CUBE_HALF, TOWER_CUBE_HALF);
        bs.add(p);
    }

    bs.upload();
    s.wait();
    BodyView bv = bs.view();

    // ── Scene descriptor + meshmap ────────────────────────────────────────────

    {
        // Shape catalog for viz: one entry per body (bodies may share shape handles).
        std::vector<uint32_t>   body_shape_idx(N_BODIES);
        std::vector<ShapeParams> body_shapes(N_BODIES);
        const std::vector<uint32_t>& bsh = bs.body_shapes();
        for (int i = 0; i < N_BODIES; ++i) {
            body_shape_idx[i] = bsh[i];
            // Reconstruct ShapeParams for each body for the viz scene descriptor.
            if (i < 9) {
                body_shapes[i] = {ShapeType::Box, PANDA_HX[i], PANDA_HY[i], PANDA_HZ[i]};
            } else if (i < N_PANDA) {
                body_shapes[i] = {ShapeType::Box, PANDA_HX[9], PANDA_HY[9], PANDA_HZ[9]};
            } else if (i == IDX_GND) {
                body_shapes[i] = {ShapeType::Box, 2.f, 2.f, 0.01f};
            } else if (i == IDX_TBL) {
                body_shapes[i] = {ShapeType::Box, 0.10f, 0.10f, TABLE_A_HZ};
            } else if (i == IDX_PICK) {
                body_shapes[i] = {ShapeType::Box, CUBE_HALF, CUBE_HALF, CUBE_HALF};
            } else {  // tower cubes
                body_shapes[i] = {ShapeType::Box, TOWER_CUBE_HALF, TOWER_CUBE_HALF, TOWER_CUBE_HALF};
            }
        }
        write_scene(prefix, N_BODIES, body_shape_idx.data(), body_shapes.data(), N_BODIES);
    }

    // Meshmap: Panda bodies → collision STL files.  Absolute paths (resolved from
    // the executable's asset dir) so the viz tools find them regardless of the cwd
    // the trajectory is replayed from.
    {
        std::vector<std::pair<uint32_t, std::string>> entries;
        const char* stl_names[] = {
            "link0","link1","link2","link3","link4",
            "link5","link6","link7","hand","finger","finger"
        };
        std::filesystem::path mesh_dir =
            std::filesystem::absolute(std::filesystem::path(asset_dir) / "meshes" / "collision");
        for (int i = 0; i < N_PANDA; ++i) {
            std::string path = (mesh_dir / (std::string(stl_names[i]) + ".stl")).string();
            entries.push_back({static_cast<uint32_t>(i), path});
        }
        write_meshmap(prefix, entries);
    }

    // ── Joints (empty — arm driven kinematically via velocity) ────────────────

    JointStore js(s, 1);
    js.upload();
    s.wait();
    JointView jv = js.view();

    // ── Physics objects ───────────────────────────────────────────────────────

    Broadphase  bp(s, N_BODIES, 4096u);
    Narrowphase np(s, 2048u);
    XpbdSolver  solver(s, 40, 8192u, 5.0f);

    IntegratorParams ip;
    ip.gravity    = {0.f, 0.f, -9.81f};   // z-up
    ip.dt         = 1.f / 60.f;
    ip.n_substeps = 1;

    const AABB bounds = {-2.f, -2.f, -0.5f, 2.f, 2.f, 3.f};

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
        q_queue.memcpy(hx.data(),  bv.pos_x, N_BODIES * sizeof(float));
        q_queue.memcpy(hy.data(),  bv.pos_y, N_BODIES * sizeof(float));
        q_queue.memcpy(hz.data(),  bv.pos_z, N_BODIES * sizeof(float));
        q_queue.memcpy(hw.data(),  bv.rot_w, N_BODIES * sizeof(float));
        q_queue.memcpy(hqx.data(), bv.rot_x, N_BODIES * sizeof(float));
        q_queue.memcpy(hqy.data(), bv.rot_y, N_BODIES * sizeof(float));
        q_queue.memcpy(hqz.data(), bv.rot_z, N_BODIES * sizeof(float)).wait();
    };

    auto write_traj_frame = [&]() {
        for (int i = 0; i < N_BODIES; ++i) {
            float row[7] = {hx[i], hy[i], hz[i], hw[i], hqx[i], hqy[i], hqz[i]};
            traj.write(reinterpret_cast<const char*>(row), sizeof(row));
        }
    };

    // Host arrays for kinematic body positions and velocities.
    // Positions are set directly on the device each frame (teleport approach),
    // since mass=0 bodies are not advanced by the integrator.
    // Velocities are set for friction computation in XPBD.
    std::vector<float> kpos_x(N_PANDA), kpos_y(N_PANDA), kpos_z(N_PANDA);
    std::vector<float> krot_w(N_PANDA), krot_x(N_PANDA), krot_y(N_PANDA), krot_z(N_PANDA);
    std::vector<float> kvel_x(N_PANDA, 0.f), kvel_y(N_PANDA, 0.f), kvel_z(N_PANDA, 0.f);
    std::vector<float> kang_x(N_PANDA, 0.f), kang_y(N_PANDA, 0.f), kang_z(N_PANDA, 0.f);

    // Initialise position cache from init_poses.
    for (int i = 0; i < N_PANDA; ++i) {
        kpos_x[i] = init_poses[i].first.x;
        kpos_y[i] = init_poses[i].first.y;
        kpos_z[i] = init_poses[i].first.z;
        krot_w[i] = init_poses[i].second.w;
        krot_x[i] = init_poses[i].second.xyz.x;
        krot_y[i] = init_poses[i].second.xyz.y;
        krot_z[i] = init_poses[i].second.xyz.z;
    }

    // ── Simulation loop ───────────────────────────────────────────────────────
    //
    // Each 60 Hz frame is integrated in N_SUB substeps.  Substepping shrinks the
    // per-step dt, which keeps contact penetration small and makes the friction
    // grasp stable through the lift and the transport swing (a single full-dt step
    // lets the gripped cube slip).  The kinematic Panda pose is interpolated across
    // the substeps so the arm moves smoothly within the frame.

    constexpr int   N_SUB    = 12;
    const float     sub_dt   = ip.dt / float(N_SUB);
    const float     inv_subdt = 1.f / sub_dt;
    constexpr int   KS = 1;          // first kinematic body
    constexpr int   KC = N_PANDA-1;  // kinematic bodies (link1..rightfinger)

    IntegratorParams ip_sub = ip;
    ip_sub.dt = sub_dt;

    auto wall_start = Clock::now();

    for (int frame = 0; frame < n_frames; ++frame) {
        for (int sub = 0; sub < N_SUB; ++sub) {
            // Fractional frame time for this substep's kinematic target.
            float ft = float(frame) + float(sub + 1) / float(N_SUB);
            float q_cur[7]; float qf_cur;
            get_targets_f(ft, q_cur, qf_cur);
            auto cur_poses = panda_fk(q_cur, qf_cur, qf_cur);

            // Teleport kinematic bodies to the interpolated FK pose; set velocities
            // (Δpose / sub_dt) for the XPBD friction computation.
            for (int i = KS; i < N_PANDA; ++i) {
                const auto& [cp, cq] = cur_poses[i];
                kvel_x[i] = (cp.x - kpos_x[i]) * inv_subdt;
                kvel_y[i] = (cp.y - kpos_y[i]) * inv_subdt;
                kvel_z[i] = (cp.z - kpos_z[i]) * inv_subdt;
                float cqx=cq.xyz.x, cqy=cq.xyz.y, cqz=cq.xyz.z;
                float pqx=krot_x[i], pqy=krot_y[i], pqz=krot_z[i], pqw=krot_w[i];
                float qd_w= cq.w*pqw + cqx*pqx + cqy*pqy + cqz*pqz;
                float qd_x= cq.w*pqx - cqx*pqw + cqy*pqz - cqz*pqy;
                float qd_y= cq.w*pqy - cqy*pqw + cqz*pqx - cqx*pqz;
                float qd_z= cq.w*pqz - cqz*pqw + cqx*pqy - cqy*pqx;
                if (qd_w < 0.f) { qd_x=-qd_x; qd_y=-qd_y; qd_z=-qd_z; }
                kang_x[i] = 2.f*qd_x*inv_subdt;
                kang_y[i] = 2.f*qd_y*inv_subdt;
                kang_z[i] = 2.f*qd_z*inv_subdt;
                kpos_x[i]=cp.x; kpos_y[i]=cp.y; kpos_z[i]=cp.z;
                krot_w[i]=cq.w; krot_x[i]=cq.xyz.x; krot_y[i]=cq.xyz.y; krot_z[i]=cq.xyz.z;
            }

            // Gentle sustained squeeze while gripping: keeps both finger-cube
            // contacts firmly and symmetrically engaged (raises the normal impulse
            // and thus the friction bound), so the cube neither drifts during the
            // lift nor slips during the transport swing.
            if (qf_cur < Q_FINGER_OPEN - 0.004f) {
                const float V_SQUEEZE = 0.0f;
                const auto& lfp = cur_poses[9].first;
                const auto& rfp = cur_poses[10].first;
                float ax = rfp.x-lfp.x, ay = rfp.y-lfp.y, az = rfp.z-lfp.z;
                float an = std::sqrt(ax*ax+ay*ay+az*az);
                if (an > 1e-6f) {
                    ax/=an; ay/=an; az/=an;
                    kvel_x[9]  += V_SQUEEZE*ax; kvel_y[9]  += V_SQUEEZE*ay; kvel_z[9]  += V_SQUEEZE*az;
                    kvel_x[10] -= V_SQUEEZE*ax; kvel_y[10] -= V_SQUEEZE*ay; kvel_z[10] -= V_SQUEEZE*az;
                }
            }

            q_queue.memcpy(bv.pos_x+KS, kpos_x.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.pos_y+KS, kpos_y.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.pos_z+KS, kpos_z.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.rot_w+KS, krot_w.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.rot_x+KS, krot_x.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.rot_y+KS, krot_y.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.rot_z+KS, krot_z.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.vel_x+KS, kvel_x.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.vel_y+KS, kvel_y.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.vel_z+KS, kvel_z.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.ang_x+KS, kang_x.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.ang_y+KS, kang_y.data()+KS, KC*sizeof(float));
            q_queue.memcpy(bv.ang_z+KS, kang_z.data()+KS, KC*sizeof(float)).wait();

            integrate(s, bv, ip_sub);
            bp.build_and_query(s, bv, sv, bounds);
            uint32_t np_ = bp.download_count(s);
            bp.sort_pairs(s, np_);
            np.run(s, bp.pairs_ptr(), np_, bv, sv, hv, mv);
            solver.solve(s, np.contacts(), jv, bv, sub_dt);
        }

        if (frame % 2 == 0) {
            download_state();
            write_traj_frame();
        }

        if (frame % 50 == 49) {
            download_state();
            spdlog::info("frame {}/{} cube=({:.3f},{:.3f},{:.3f}) lf=({:.3f},{:.3f},{:.3f})",
                frame + 1, n_frames,
                hx[IDX_PICK], hy[IDX_PICK], hz[IDX_PICK], hx[9], hy[9], hz[9]);
        }
    }
    s.wait();
    traj.close();

    double wall_sec = Seconds(Clock::now() - wall_start).count();
    double fps      = n_frames / wall_sec;

    // ── Validate: cube should be on top of tower ──────────────────────────────

    download_state();

    // A successful place: the picked cube rests on the tower top — within the tower
    // footprint in xy AND sitting above the tower's top surface.
    float cube_fx = hx[IDX_PICK], cube_fy = hy[IDX_PICK], cube_fz = hz[IDX_PICK];
    float dxy = std::hypot(cube_fx - B_X, cube_fy - B_Y);
    float top_existing_z = TOWER_TOP;  // tower top surface
    bool  placed = (dxy < TOWER_CUBE_HALF) && (cube_fz > top_existing_z);
    if (!placed) {
        spdlog::warn("Cube did not land on tower: final=({:.3f},{:.3f},{:.3f}) "
                     "tower xy=({:.2f},{:.2f}) dxy={:.3f} (need <{:.3f}, z>{:.3f})",
                     cube_fx, cube_fy, cube_fz, B_X, B_Y, dxy,
                     2.f * CUBE_HALF, top_existing_z + CUBE_HALF);
    } else {
        spdlog::info("Cube placed on tower: final=({:.3f},{:.3f},{:.3f})",
                     cube_fx, cube_fy, cube_fz);
    }
    float cube_final_z = cube_fz;

    // ── Final-state hash ──────────────────────────────────────────────────────

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
            "  \"n_panda_bodies\": {},\n"
            "  \"n_frames\": {},\n"
            "  \"simulation_seconds\": {:.4f},\n"
            "  \"wall_clock_seconds\": {:.4f},\n"
            "  \"frames_per_second\": {:.2f},\n"
            "  \"realtime_factor\": {:.3f},\n"
            "  \"cube_placed\": {},\n"
            "  \"cube_final_z\": {:.4f}\n"
            "}}\n",
            N_BODIES, N_PANDA, n_frames,
            n_frames * static_cast<double>(ip.dt),
            wall_sec, fps, fps / 60.0,
            placed ? "true" : "false",
            cube_final_z
        );
    }

    spdlog::info("Device: {} ({})", dev.name(), dev.is_gpu() ? "GPU" : "CPU");
    spdlog::info("Completed {} frames in {:.3f}s  ({:.1f} fps, {:.2f}x realtime)",
                 n_frames, wall_sec, fps, fps / 60.0);
    spdlog::info("Hash: {:016x}", hash);

    return 0;
}
