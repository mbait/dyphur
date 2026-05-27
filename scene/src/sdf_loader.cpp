#include <scene/sdf_loader.hpp>
#include <core/body.hpp>
#include <core/articulation.hpp>
#include <core/math/math.hpp>

#include <sdf/Root.hh>
#include <sdf/World.hh>
#include <sdf/Model.hh>
#include <sdf/Link.hh>
#include <sdf/Joint.hh>
#include <sdf/JointAxis.hh>
#include <sdf/Collision.hh>
#include <sdf/Geometry.hh>
#include <sdf/Box.hh>
#include <sdf/Sphere.hh>
#include <sdf/Cylinder.hh>
#include <sdf/Mesh.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Inertial.hh>

#include <algorithm>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>

namespace dyphur {

// ── Simple OBJ mesh loader ────────────────────────────────────────────────────
// Handles the subset used by typical robot/scene meshes:
//   v x y z
//   f i j k  (1-indexed, simple triangles only)
static VertexBuffer load_obj(const std::string& path, float scale)
{
    std::ifstream f(path);
    if (!f) throw std::runtime_error("load_obj: cannot open " + path);

    VertexBuffer vb;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        if (line[0] == 'v' && line[1] == ' ') {
            float x, y, z;
            if (std::sscanf(line.c_str(), "v %f %f %f", &x, &y, &z) == 3) {
                vb.x.push_back(x * scale);
                vb.y.push_back(y * scale);
                vb.z.push_back(z * scale);
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            unsigned int a, b, c;
            // Handle "f v/vt/vn" and "f v//vn" and plain "f v"
            int ia=0, ib=0, ic=0;
            if (std::sscanf(line.c_str(), "f %d/%*d/%*d %d/%*d/%*d %d/%*d/%*d", &ia,&ib,&ic)==3 ||
                std::sscanf(line.c_str(), "f %d//%*d %d//%*d %d//%*d", &ia,&ib,&ic)==3 ||
                std::sscanf(line.c_str(), "f %d %d %d", &ia,&ib,&ic)==3) {
                a = static_cast<uint32_t>(ia-1);
                b = static_cast<uint32_t>(ib-1);
                c = static_cast<uint32_t>(ic-1);
                vb.idx_a.push_back(a);
                vb.idx_b.push_back(b);
                vb.idx_c.push_back(c);
            }
        }
    }
    if (vb.x.empty()) throw std::runtime_error("load_obj: no vertices in " + path);
    return vb;
}

// Resolve a mesh URI relative to the SDF file's directory.
static std::string resolve_uri(const std::string& uri, const std::string& sdf_dir)
{
    // Strip common prefixes
    std::string path = uri;
    const std::string file_prefix = "file://";
    const std::string model_prefix = "model://";
    if (path.rfind(file_prefix, 0) == 0) path = path.substr(file_prefix.size());
    else if (path.rfind(model_prefix, 0) == 0) path = path.substr(model_prefix.size());

    if (!path.empty() && path[0] != '/') path = sdf_dir + "/" + path;
    return path;
}

// ── Pose helpers ──────────────────────────────────────────────────────────────
static void gz_pose_to_body(const gz::math::Pose3d& pose, BodyParams& bp, float scale)
{
    bp.position.x = static_cast<float>(pose.Pos().X()) * scale;
    bp.position.y = static_cast<float>(pose.Pos().Y()) * scale;
    bp.position.z = static_cast<float>(pose.Pos().Z()) * scale;

    auto q = pose.Rot();
    // gz::math quaternion: W, X, Y, Z
    bp.rotation = { static_cast<float>(q.W()),
                    static_cast<float>(q.X()),
                    static_cast<float>(q.Y()),
                    static_cast<float>(q.Z()) };
}

// ── Joint type mapping ────────────────────────────────────────────────────────
static JointType sdf_joint_type(sdf::JointType t)
{
    switch (t) {
        case sdf::JointType::FIXED:     return JointType::Fixed;
        case sdf::JointType::REVOLUTE:  return JointType::Revolute;
        case sdf::JointType::PRISMATIC: return JointType::Prismatic;
        case sdf::JointType::BALL:      return JointType::Ball;
        default:                         return JointType::Fixed; // best approximation
    }
}

// ── Main loader ───────────────────────────────────────────────────────────────
SceneDesc load_sdf(const std::string& path, const SdfLoadParams& p)
{
    sdf::Root root;
    sdf::Errors errors = root.Load(path);
    for (const auto& e : errors)
        if (e.Code() != sdf::ErrorCode::NONE)
            throw std::runtime_error("load_sdf: " + e.Message());

    // Determine directory for relative mesh URI resolution
    std::string sdf_dir = ".";
    auto slash = path.rfind('/');
    if (slash != std::string::npos) sdf_dir = path.substr(0, slash);

    SceneDesc scene;

    // Helper: name → body index map and name → world pose map
    std::map<std::string, uint32_t>         name_to_idx;
    std::map<std::string, gz::math::Pose3d> name_to_world_pose;

    auto process_model = [&](const sdf::Model* model) {
        std::string model_name = model->Name();

        // Model's pose in the world frame: use RawPose (pose in parent = world).
        // SemanticPose().Resolve("world") silently fails when the libsdformat
        // pose graph doesn't have a registered "world" frame node, returning
        // identity. RawPose() always returns the literal <pose> from the SDF.
        gz::math::Pose3d model_world_pose = model->RawPose();

        for (uint64_t li = 0; li < model->LinkCount(); ++li) {
            const sdf::Link* link = model->LinkByIndex(li);
            std::string link_name = model_name + "::" + link->Name();

            // Link pose in model frame, composed with model-in-world pose.
            gz::math::Pose3d link_world_pose = model_world_pose * link->RawPose();
            name_to_world_pose[link_name] = link_world_pose;

            BodyDesc bd;
            bd.name = link_name;
            gz_pose_to_body(link_world_pose, bd.body, p.scale);

            // Mass / inertia
            bool is_static = model->Static();
            if (is_static) {
                bd.body.mass = 0.f;
                bd.body.flags = BodyFlag::Static;
                bd.body.inertia = Mat3f::identity();
            } else {
                gz::math::Inertiald inertial = link->Inertial();
                bd.body.mass = static_cast<float>(inertial.MassMatrix().Mass());
                if (bd.body.mass <= 0.f) bd.body.mass = 1.f;
                auto ixx = inertial.MassMatrix().Ixx();
                auto iyy = inertial.MassMatrix().Iyy();
                auto izz = inertial.MassMatrix().Izz();
                float fx = static_cast<float>(ixx);
                float fy = static_cast<float>(iyy);
                float fz = static_cast<float>(izz);
                bd.body.inertia = Mat3f(fx,0,0, 0,fy,0, 0,0,fz);
                bd.body.flags = 0;
            }

            // Collision geometry (use first collision)
            bool shape_set = false;
            for (uint64_t ci = 0; ci < link->CollisionCount() && !shape_set; ++ci) {
                const sdf::Collision* coll = link->CollisionByIndex(ci);
                const sdf::Geometry* geom  = coll->Geom();

                switch (geom->Type()) {
                    case sdf::GeometryType::BOX: {
                        const sdf::Box* box = geom->BoxShape();
                        bd.shape.type   = ShapeType::Box;
                        bd.shape.half_x = static_cast<float>(box->Size().X() / 2.0) * p.scale;
                        bd.shape.half_y = static_cast<float>(box->Size().Y() / 2.0) * p.scale;
                        bd.shape.half_z = static_cast<float>(box->Size().Z() / 2.0) * p.scale;
                        shape_set = true;
                        break;
                    }
                    case sdf::GeometryType::SPHERE: {
                        const sdf::Sphere* sp = geom->SphereShape();
                        bd.shape.type   = ShapeType::Sphere;
                        bd.shape.half_x = static_cast<float>(sp->Radius()) * p.scale;
                        shape_set = true;
                        break;
                    }
                    case sdf::GeometryType::CYLINDER: {
                        const sdf::Cylinder* cyl = geom->CylinderShape();
                        // Approximate cylinder as Box
                        bd.shape.type   = ShapeType::Box;
                        float r = static_cast<float>(cyl->Radius()) * p.scale;
                        float h = static_cast<float>(cyl->Length() / 2.0) * p.scale;
                        bd.shape.half_x = r;
                        bd.shape.half_y = h;
                        bd.shape.half_z = r;
                        shape_set = true;
                        break;
                    }
                    case sdf::GeometryType::MESH: {
                        const sdf::Mesh* mesh = geom->MeshShape();
                        std::string uri  = mesh->Uri();
                        float mscale = static_cast<float>(mesh->Scale().X()) * p.scale;
                        std::string mesh_path = resolve_uri(uri, sdf_dir);

                        // Only OBJ supported currently
                        VertexBuffer vb = load_obj(mesh_path, mscale);

                        if (is_static && p.static_mesh_as_trimesh) {
                            bd.mesh_idx = static_cast<uint32_t>(scene.meshes.size());
                            scene.meshes.push_back(std::move(vb));
                            bd.shape.type   = ShapeType::TriangleMesh;
                            bd.shape.ext_id = bd.mesh_idx;
                        } else {
                            // V-HACD decomposition
                            auto hulls = decompose_vhacd(vb, p.decomp);
                            if (!hulls.empty()) {
                                // For now, use first hull as representative shape.
                                // Multi-hull support: one body per hull (deferred).
                                bd.hull_idx = static_cast<uint32_t>(scene.hulls.size());
                                scene.hulls.push_back(hulls[0]);
                                bd.shape.type   = ShapeType::ConvexHull;
                                bd.shape.ext_id = bd.hull_idx;
                            } else {
                                // Fallback to bounding box
                                bd.shape.type = ShapeType::Box;
                                bd.shape.half_x = bd.shape.half_y = bd.shape.half_z = 0.5f;
                            }
                        }
                        shape_set = true;
                        break;
                    }
                    default:
                        // Unknown geometry: use unit box
                        bd.shape.type = ShapeType::Box;
                        bd.shape.half_x = bd.shape.half_y = bd.shape.half_z = 0.5f * p.scale;
                        shape_set = true;
                        break;
                }
            }

            if (!shape_set) {
                bd.shape.type = ShapeType::Box;
                bd.shape.half_x = bd.shape.half_y = bd.shape.half_z = 0.5f * p.scale;
            }

            name_to_idx[link_name] = static_cast<uint32_t>(scene.bodies.size());
            scene.bodies.push_back(std::move(bd));
        }

        // Joints
        for (uint64_t ji = 0; ji < model->JointCount(); ++ji) {
            const sdf::Joint* joint = model->JointByIndex(ji);
            std::string parent_full = model_name + "::" + joint->ParentName();
            std::string child_full  = model_name + "::" + joint->ChildName();

            if (name_to_idx.find(parent_full) == name_to_idx.end()) continue;
            if (name_to_idx.find(child_full)  == name_to_idx.end()) continue;

            uint32_t pi = name_to_idx[parent_full];
            uint32_t ci = name_to_idx[child_full];

            JointParams jp;
            jp.body_parent = pi;
            jp.body_child  = ci;
            jp.type        = sdf_joint_type(joint->Type());

            // Joint axis
            if (const sdf::JointAxis* axis = joint->Axis()) {
                gz::math::Vector3d av = axis->Xyz();
                jp.axis = Vec3f{static_cast<float>(av.X()),
                                static_cast<float>(av.Y()),
                                static_cast<float>(av.Z())};
                jp.limit_lo = static_cast<float>(axis->Lower());
                jp.limit_hi = static_cast<float>(axis->Upper());
                if (jp.limit_lo >= jp.limit_hi) {
                    jp.limit_lo = -1e10f; jp.limit_hi = 1e10f;
                }
            } else {
                jp.axis     = Vec3f{0.f, 0.f, 1.f};
                jp.limit_lo = -1e10f; jp.limit_hi = 1e10f;
            }

            // Anchor: joint origin in parent body frame.
            // In SDF 1.8+ the joint <pose> is in the child link frame by default.
            // Compute joint world pos = child_world * joint_raw_pose, then express
            // that in the parent body frame.
            gz::math::Pose3d child_world_pose;
            gz::math::Pose3d parent_world_pose;
            if (name_to_world_pose.count(child_full))
                child_world_pose  = name_to_world_pose.at(child_full);
            if (name_to_world_pose.count(parent_full))
                parent_world_pose = name_to_world_pose.at(parent_full);

            gz::math::Pose3d joint_world_pose = child_world_pose * joint->RawPose();
            gz::math::Pose3d anchor_in_parent  = parent_world_pose.Inverse() * joint_world_pose;

            jp.anchor_parent = Vec3f{
                static_cast<float>(anchor_in_parent.Pos().X()) * p.scale,
                static_cast<float>(anchor_in_parent.Pos().Y()) * p.scale,
                static_cast<float>(anchor_in_parent.Pos().Z()) * p.scale};
            jp.anchor_child = Vec3f{0.f, 0.f, 0.f};

            jp.stiffness     = 0.f;
            jp.damping       = 0.f;
            jp.target_pos    = 0.f;
            jp.target_vel    = 0.f;
            jp.compliance_pos = 0.f;
            jp.compliance_ang = 0.f;

            JointDesc jd;
            jd.joint       = jp;
            jd.parent_name = parent_full;
            jd.child_name  = child_full;
            scene.joints.push_back(std::move(jd));
        }
    };

    // Process world models
    if (root.WorldCount() > 0) {
        const sdf::World* world = root.WorldByIndex(0);
        for (uint64_t mi = 0; mi < world->ModelCount(); ++mi)
            process_model(world->ModelByIndex(mi));
    } else if (const sdf::Model* standalone = root.Model()) {
        // Stand-alone model SDF
        process_model(standalone);
    }

    return scene;
}

} // namespace dyphur
