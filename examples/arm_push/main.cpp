// Phase 3 exit-criterion demo: arm pushing boxes loaded from SDF.
//
// Loads arm_room.sdf (static floor + 3 pushable boxes + 1-DOF planar arm),
// applies PD control with a sinusoidal target that sweeps the arm ±90°,
// runs full broadphase + narrowphase + XPBD each frame so the arm actually
// contacts and pushes the boxes.
//
// Emits sdf_arm_push.metrics.json and a determinism hash on stdout.
//
// Usage: arm_push [world.sdf] [n_frames]

#include <scene/sdf_loader.hpp>
#include <core/body_store.hpp>
#include <core/shape_store.hpp>
#include <core/convex_hull_store.hpp>
#include <core/mesh_bvh.hpp>
#include <core/broadphase.hpp>
#include <core/narrowphase.hpp>
#include <core/integrator.hpp>
#include <core/xpbd_solver.hpp>
#include <core/contact_store.hpp>
#include <core/joint_store.hpp>
#include <compute/device.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

using namespace dyphur;

static uint64_t fnv1a64(uint64_t h, const void* data, size_t n)
{
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x00000100000001B3ULL; }
    return h;
}

int main(int argc, char** argv)
{
    const char* sdf_path = nullptr;
    int n_frames = 600;  // 10 s at 60 Hz

    if (argc >= 2) sdf_path = argv[1];
    if (argc >= 3) n_frames = std::atoi(argv[2]);

    std::string default_path;
    if (!sdf_path) {
        std::filesystem::path exe(argv[0]);
        default_path = (exe.parent_path() / "arm_room.sdf").string();
        sdf_path = default_path.c_str();
    }

    std::printf("arm_push: loading %s\n", sdf_path);

    // ── Load scene ────────────────────────────────────────────────────────────
    SdfLoadParams lp;
    lp.static_mesh_as_trimesh = false;  // no mesh geometry in this scene
    SceneDesc scene;
    try {
        scene = load_sdf(sdf_path, lp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load_sdf: %s\n", e.what());
        return 1;
    }

    const uint32_t N = static_cast<uint32_t>(scene.bodies.size());
    std::printf("arm_push: %u bodies, %zu joints, %zu hulls, %zu meshes\n",
                N, scene.joints.size(), scene.hulls.size(), scene.meshes.size());

    if (scene.joints.empty()) {
        std::fprintf(stderr, "arm_push: no joints loaded — check SDF\n");
        return 1;
    }

    // The arm base is a heavy anchor in the SDF (mass=1000 kg, dynamic model)
    // to satisfy SDF schema (joints must be within one model).  Mark it static
    // post-load so it never moves and the joint anchor is world-fixed.
    for (auto& bd : scene.bodies) {
        if (bd.name == "arm::base") {
            bd.body.mass   = 0.f;
            bd.body.flags  = BodyFlag::Static;
            bd.body.inertia = Mat3f::identity();
        }
    }

    // ── Set PD gains (loader leaves them at 0) ────────────────────────────────
    for (auto& jd : scene.joints) {
        if (jd.joint.type == JointType::Revolute) {
            jd.joint.stiffness      = 400.f;
            jd.joint.damping        = 30.f;
            // Small compliance prevents solver divergence when joint and
            // contact constraints compete (e.g. arm base resting on floor).
            jd.joint.compliance_pos = 1e-5f;
            jd.joint.compliance_ang = 1e-5f;
        }
    }

    // ── Compute ───────────────────────────────────────────────────────────────
    Device dev = Device::default_cpu();
    auto   s   = dev.make_stream();
    auto&  q   = s.queue();

    // ── Shapes ───────────────────────────────────────────────────────────────
    ShapeStore ss(s, N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t h = ss.add(scene.bodies[i].shape);
        scene.bodies[i].body.shape_handle = h;
    }
    ss.upload();
    ShapeView sv = ss.view();

    // ── Convex hull catalog (empty for this scene) ────────────────────────────
    ConvexHullStore hs(s, static_cast<uint32_t>(scene.hulls.size() + 1), 4096);
    for (const auto& hull : scene.hulls)
        hs.add(hull.x.data(), hull.y.data(), hull.z.data(),
               static_cast<uint32_t>(hull.x.size()));
    if (!scene.hulls.empty()) hs.upload();
    ConvexHullView hv = hs.view();

    // ── Mesh BVH catalog (empty for this scene) ───────────────────────────────
    MeshBvhStore ms(s,
        static_cast<uint32_t>(scene.meshes.size() + 1),
        65536, 65536, 131072);
    for (const auto& mesh : scene.meshes) {
        MeshBvh bvh;
        bvh.build(mesh.x.data(), mesh.y.data(), mesh.z.data(),
                  static_cast<uint32_t>(mesh.x.size()),
                  mesh.idx_a.data(), mesh.idx_b.data(), mesh.idx_c.data(),
                  static_cast<uint32_t>(mesh.idx_a.size()));
        ms.add(bvh);
    }
    if (!scene.meshes.empty()) ms.upload();
    MeshBvhCatalogView mv = ms.view();


    // ── Bodies ────────────────────────────────────────────────────────────────
    BodyStore bs(s, N);
    for (const auto& bd : scene.bodies) bs.add(bd.body);
    bs.upload(); s.wait();
    BodyView bv = bs.view();

    // ── Joints ────────────────────────────────────────────────────────────────
    const uint32_t NJ = static_cast<uint32_t>(scene.joints.size());
    JointStore js(s, NJ + 1);
    for (const auto& jd : scene.joints) js.add(jd.joint);
    js.upload(); s.wait();
    JointView jv = js.view();

    // ── Physics objects ───────────────────────────────────────────────────────
    Broadphase  bp(s, N, 8192u);
    Narrowphase np(s, 4096u);
    XpbdSolver  solver(s, 20);
    ContactStore cs(s, 1); cs.reset(s); s.wait();

    IntegratorParams ip;
    ip.gravity    = {0.f, 0.f, -9.81f};
    ip.dt         = 1.f / 60.f;
    ip.n_substeps = 1;

    const AABB bounds = {-3.f, -3.f, -1.f, 3.f, 3.f, 3.f};

    // PD target: sinusoidal sweep ±1.55 rad (≈ ±89°), just under the ±π/2 limit.
    // At θ = ±84° the arm side first touches box_a/box_b (y=±0.75 m).
    // Period = 6 s → arm makes ~1.7 full sweeps in 600 frames (10 s).
    const float AMP   = 1.55f;          // radians  (< π/2 = 1.5708)
    const float OMEGA = 3.14159f / 3.f; // rad/s  → period 6 s

    // ── Simulation loop ───────────────────────────────────────────────────────
    auto t0 = std::chrono::steady_clock::now();
    float sim_t = 0.f;

    for (int f = 0; f < n_frames; ++f) {
        // Sinusoidal target for all revolute joints (only j0 here).
        float target_pos = AMP * std::sin(OMEGA * sim_t);
        float target_vel = 0.f;
        js.set_targets(&target_pos, &target_vel);

        integrate(s, bv, ip);
        bp.build_and_query(s, bv, sv, bounds);
        uint32_t np_ = bp.download_count(s);  // one sync per frame
        bp.sort_pairs(s, np_);                 // no-op when np_ <= 1
        np.run(s, bp.pairs_ptr(), np_, bv, sv, hv, mv);  // always resets contact store
        solver.solve(s, np.contacts(), jv, bv, ip.dt);

        sim_t += ip.dt;
    }
    s.wait();  // single end-of-batch sync for timing

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double fps     = n_frames / elapsed;

    // ── Hash final state ──────────────────────────────────────────────────────
    std::vector<float> hx(N), hy(N), hz(N);
    q.memcpy(hx.data(), bv.pos_x, N * sizeof(float));
    q.memcpy(hy.data(), bv.pos_y, N * sizeof(float));
    q.memcpy(hz.data(), bv.pos_z, N * sizeof(float)).wait();

    constexpr uint64_t FNV_OFFSET = 0xcbf29ce484222325ULL;
    uint64_t h = FNV_OFFSET;
    h = fnv1a64(h, hx.data(), N * sizeof(float));
    h = fnv1a64(h, hy.data(), N * sizeof(float));
    h = fnv1a64(h, hz.data(), N * sizeof(float));

    // ── Output ────────────────────────────────────────────────────────────────
    std::printf("arm_push: %d frames in %.3f s  →  %.1f fps  (%.2fx realtime)\n",
                n_frames, elapsed, fps, fps / 60.0);
    std::printf("arm_push: final state hash = 0x%016llx\n",
                static_cast<unsigned long long>(h));

    // Verify boxes moved (arm pushed at least one)
    std::printf("arm_push: box positions after %d frames:\n", n_frames);
    // Bodies: [0]=floor, [1]=box_a, [2]=box_b, [3]=arm::base, [4]=arm::link1
    for (uint32_t i = 0; i < N; ++i)
        std::printf("  body[%u] %s: (%.4f, %.4f, %.4f)\n",
            i, scene.bodies[i].name.c_str(), hx[i], hy[i], hz[i]);

    std::FILE* mf = std::fopen("sdf_arm_push.metrics.json", "w");
    if (mf) {
        std::fprintf(mf,
            "{\n"
            "  \"n_bodies\": %u,\n"
            "  \"n_joints\": %u,\n"
            "  \"n_frames\": %d,\n"
            "  \"elapsed_s\": %.4f,\n"
            "  \"fps\": %.2f,\n"
            "  \"realtime_factor\": %.2f,\n"
            "  \"final_hash\": \"0x%016llx\"\n"
            "}\n",
            N, NJ, n_frames, elapsed, fps, fps / 60.0,
            static_cast<unsigned long long>(h));
        std::fclose(mf);
        std::printf("arm_push: metrics written to sdf_arm_push.metrics.json\n");
    }

    return 0;
}
