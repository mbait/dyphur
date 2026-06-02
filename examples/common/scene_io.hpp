#pragma once
#include <core/shapes.hpp>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dyphur {

// On-disk scene descriptor: maps bodies to shapes with geometry info.
// Written alongside .trajectory files so the visualiser can reconstruct geometry.
struct SceneFileDesc {
    uint32_t              n_bodies = 0;
    uint32_t              n_shapes = 0;
    std::vector<uint32_t> body_shape_idx; // [n_bodies] — index into shapes[]
    std::vector<ShapeParams> shapes;       // [n_shapes]
};

// Binary .scene format:
//   uint32_t n_bodies, n_shapes
//   uint32_t body_shape_idx[n_bodies]
//   per shape (20 bytes): uint32_t type, float half_x/y/z, uint32_t _pad
//
// ConvexHull/TriangleMesh entries render as their bounding box in the visualiser.

inline void write_scene(const std::string& prefix,
                        uint32_t n_bodies,
                        const uint32_t* body_shape_idx,
                        const ShapeParams* shapes,
                        uint32_t n_shapes) {
    std::ofstream f(prefix + ".scene", std::ios::binary);
    if (!f) throw std::runtime_error("write_scene: cannot open " + prefix + ".scene");

    uint32_t hdr[2] = {n_bodies, n_shapes};
    f.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
    f.write(reinterpret_cast<const char*>(body_shape_idx),
            n_bodies * sizeof(uint32_t));

    for (uint32_t i = 0; i < n_shapes; ++i) {
        uint32_t type = static_cast<uint32_t>(shapes[i].type);
        float hx = shapes[i].half_x, hy = shapes[i].half_y, hz = shapes[i].half_z;
        uint32_t pad = 0;
        f.write(reinterpret_cast<const char*>(&type), 4);
        f.write(reinterpret_cast<const char*>(&hx),   4);
        f.write(reinterpret_cast<const char*>(&hy),   4);
        f.write(reinterpret_cast<const char*>(&hz),   4);
        f.write(reinterpret_cast<const char*>(&pad),  4);
    }
}

inline SceneFileDesc read_scene(const std::string& prefix) {
    std::ifstream f(prefix + ".scene", std::ios::binary);
    if (!f) throw std::runtime_error("read_scene: cannot open " + prefix + ".scene");

    SceneFileDesc desc;
    uint32_t hdr[2];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    desc.n_bodies = hdr[0];
    desc.n_shapes = hdr[1];

    desc.body_shape_idx.resize(desc.n_bodies);
    f.read(reinterpret_cast<char*>(desc.body_shape_idx.data()),
           desc.n_bodies * sizeof(uint32_t));

    desc.shapes.resize(desc.n_shapes);
    for (uint32_t i = 0; i < desc.n_shapes; ++i) {
        uint32_t type, pad;
        float hx, hy, hz;
        f.read(reinterpret_cast<char*>(&type), 4);
        f.read(reinterpret_cast<char*>(&hx),   4);
        f.read(reinterpret_cast<char*>(&hy),   4);
        f.read(reinterpret_cast<char*>(&hz),   4);
        f.read(reinterpret_cast<char*>(&pad),  4);
        desc.shapes[i].type   = static_cast<ShapeType>(type);
        desc.shapes[i].half_x = hx;
        desc.shapes[i].half_y = hy;
        desc.shapes[i].half_z = hz;
    }

    return desc;
}

// ── Mesh map sidecar ──────────────────────────────────────────────────────────
// <prefix>.meshmap: plain text, one entry per line: "<body_idx> <path>"
// Paths are relative to the directory of the executable that writes the file.
// Viz tools use this to render bodies with their actual mesh geometry instead
// of the fallback box/sphere. Absent = all bodies use shape-based rendering.

inline void write_meshmap(
    const std::string& prefix,
    const std::vector<std::pair<uint32_t, std::string>>& entries)
{
    std::ofstream f(prefix + ".meshmap");
    if (!f) throw std::runtime_error("write_meshmap: cannot open " + prefix + ".meshmap");
    for (const auto& [idx, path] : entries)
        f << idx << " " << path << "\n";
}

inline std::unordered_map<uint32_t, std::string>
read_meshmap(const std::string& prefix)
{
    std::unordered_map<uint32_t, std::string> result;
    std::ifstream f(prefix + ".meshmap");
    if (!f) return result;  // absent is fine — caller falls back to box/sphere
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        uint32_t idx;
        std::string path;
        if (ss >> idx >> path)
            result[idx] = path;
    }
    return result;
}

// Trajectory file header (simulation frame count, not trajectory frame count;
// trajectory frames = every-other simulation frame = ceil(n_frames / 2)).
struct TrajHeader {
    uint32_t n_bodies;
    uint32_t n_frames; // simulation frames
};

inline TrajHeader read_traj_header(const std::string& prefix) {
    std::ifstream f(prefix + ".trajectory", std::ios::binary);
    if (!f) throw std::runtime_error("read_traj_header: cannot open " + prefix + ".trajectory");
    TrajHeader h{};
    f.read(reinterpret_cast<char*>(&h), sizeof(h));
    return h;
}

} // namespace dyphur
