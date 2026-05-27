// SDF scene demo — Phase 3 exit-criterion example.
//
// Loads a small SDF world (box_room.sdf: static floor + 2 dynamic boxes),
// simulates N frames, and emits a metrics JSON + determinism hash.
//
// Gravity is z-down to match SDF native (z-up) coordinate frame.
//
// Usage:  sdf_scene [world.sdf] [N_frames]
//         sdf_scene              → box_room.sdf in same directory, 240 frames

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
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <vector>

using namespace dyphur;

static uint64_t fnv1a64(uint64_t h, const void* data, size_t n) {
    const auto* p = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 0x00000100000001B3ULL; }
    return h;
}

int main(int argc, char** argv)
{
    // ── CLI ───────────────────────────────────────────────────────────────────
    const char* sdf_path  = nullptr;
    int n_frames = 240;

    if (argc >= 2) sdf_path = argv[1];
    if (argc >= 3) n_frames = std::atoi(argv[2]);

    // Default: box_room.sdf next to this binary
    std::string default_path;
    if (!sdf_path) {
        // Try same dir as argv[0]
        std::filesystem::path exe(argv[0]);
        default_path = (exe.parent_path() / "box_room.sdf").string();
        // Fallback: next to source
        if (!std::filesystem::exists(default_path))
            default_path = std::string(__FILE__);
        sdf_path = default_path.c_str();
    }

    std::printf("sdf_scene: loading %s\n", sdf_path);

    // ── Load SDF ──────────────────────────────────────────────────────────────
    SdfLoadParams lp;
    lp.static_mesh_as_trimesh = true;
    SceneDesc scene;
    try {
        scene = load_sdf(sdf_path, lp);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "load_sdf error: %s\n", e.what());
        return 1;
    }

    std::printf("sdf_scene: %zu bodies, %zu joints, %zu hulls, %zu meshes\n",
                scene.bodies.size(), scene.joints.size(),
                scene.hulls.size(), scene.meshes.size());

    // ── Setup compute ─────────────────────────────────────────────────────────
    Device dev = Device::default_cpu();
    auto   s   = dev.make_stream();
    auto&  q   = s.queue();

    const uint32_t N = static_cast<uint32_t>(scene.bodies.size());

    // ── Upload shapes ─────────────────────────────────────────────────────────
    ShapeStore ss(s, N);
    for (uint32_t i = 0; i < N; ++i) {
        uint32_t h = ss.add(scene.bodies[i].shape);
        scene.bodies[i].body.shape_handle = h;
    }
    ss.upload();
    ShapeView sv = ss.view();

    // ── Convex hull catalog ───────────────────────────────────────────────────
    ConvexHullStore hs(s, static_cast<uint32_t>(scene.hulls.size() + 1), 4096);
    for (const auto& hull : scene.hulls)
        hs.add(hull.x.data(), hull.y.data(), hull.z.data(),
               static_cast<uint32_t>(hull.x.size()));
    if (!scene.hulls.empty()) hs.upload();
    ConvexHullView hv = hs.view();

    // ── Mesh BVH catalog ──────────────────────────────────────────────────────
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

    // ── Upload bodies ─────────────────────────────────────────────────────────
    BodyStore bs(s, N);
    for (auto& bd : scene.bodies) bs.add(bd.body);
    bs.upload(); s.wait();
    BodyView bv = bs.view();

    // ── Joints ────────────────────────────────────────────────────────────────
    JointStore js(s, static_cast<uint32_t>(scene.joints.size() + 1));
    for (const auto& jd : scene.joints) js.add(jd.joint);
    if (!scene.joints.empty()) { js.upload(); s.wait(); }
    JointView jv = js.view();

    // ── Broadphase / narrowphase / solver ─────────────────────────────────────
    Broadphase  bp(s, N, 4096u);
    Narrowphase np(s, 2048u);
    XpbdSolver  solver(s, 10);

    ContactStore cs(s, 1); cs.reset(s); s.wait();

    IntegratorParams ip;
    ip.gravity    = {0.f, 0.f, -9.81f}; // z-down (SDF native frame)
    ip.dt         = 1.f / 60.f;
    ip.n_substeps = 1;

    const AABB bounds = {-20, -20, -5, 20, 20, 30};

    // ── Simulation loop ───────────────────────────────────────────────────────
    auto t0 = std::chrono::steady_clock::now();

    for (int f = 0; f < n_frames; ++f) {
        integrate(s, bv, ip); s.wait();
        bp.build_and_query(s, bv, sv, bounds); s.wait();
        uint32_t np_ = bp.download_count(s);
        bp.sort_pairs(s, np_); s.wait();
        np.run(s, bp.pairs_ptr(), np_, bv, sv, hv, mv); s.wait();
        solver.solve(s, np.contacts(), jv, bv, ip.dt); s.wait();
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    double fps = n_frames / elapsed;

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
    std::printf("sdf_scene: %d frames in %.3f s  →  %.1f fps  (%.2fx realtime)\n",
                n_frames, elapsed, fps, fps / 60.0);
    std::printf("sdf_scene: final state hash = 0x%016llx\n",
                static_cast<unsigned long long>(h));

    // Metrics JSON
    std::FILE* mf = std::fopen("sdf_scene.metrics.json", "w");
    if (mf) {
        std::fprintf(mf,
            "{\n"
            "  \"n_bodies\": %u,\n"
            "  \"n_frames\": %d,\n"
            "  \"elapsed_s\": %.4f,\n"
            "  \"fps\": %.2f,\n"
            "  \"realtime_factor\": %.2f,\n"
            "  \"final_hash\": \"0x%016llx\"\n"
            "}\n",
            N, n_frames, elapsed, fps, fps / 60.0,
            static_cast<unsigned long long>(h));
        std::fclose(mf);
        std::printf("sdf_scene: metrics written to sdf_scene.metrics.json\n");
    }

    return 0;
}
