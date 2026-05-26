#pragma once
#include "math/math.hpp"
#include <cstdint>

namespace dyphur {

enum class JointType : uint8_t {
    Fixed     = 0,  // 0 DOF: no relative motion
    Revolute  = 1,  // 1 DOF: rotation about axis
    Prismatic = 2,  // 1 DOF: translation along axis
    Ball      = 3,  // 3 DOF: free rotation (spherical)
};

// Host-facing joint description. Passed to JointStore::add().
struct JointParams {
    uint32_t  body_parent   = 0;
    uint32_t  body_child    = 0;

    // Anchor point in parent and child body frames (local coordinates).
    Vec3f     anchor_parent = Vec3f::zero();
    Vec3f     anchor_child  = Vec3f::zero();

    // Rotation/translation axis in parent body frame (normalized).
    // Used by Revolute and Prismatic.
    Vec3f     axis          = {0.f, 1.f, 0.f};

    // Joint limits [lo, hi]. Units: radians for Revolute, meters for Prismatic.
    // lo > hi disables limits.
    float     limit_lo      =  1.f;
    float     limit_hi      = -1.f;

    // PD motor targets and gains. Applied as soft constraints.
    float     target_pos    = 0.f;
    float     target_vel    = 0.f;
    float     stiffness     = 0.f;
    float     damping       = 0.f;

    // XPBD compliance (inverse of stiffness). Smaller = stiffer.
    float     compliance_pos = 0.f;
    float     compliance_ang = 0.f;

    JointType type          = JointType::Fixed;
};

// Kernel-facing SoA view: raw device pointers + active joint count.
// Trivially copyable; safe to capture in a kernel lambda.
struct JointView {
    uint32_t* body_parent;
    uint32_t* body_child;

    float* anchor_px;  float* anchor_py;  float* anchor_pz;
    float* anchor_cx;  float* anchor_cy;  float* anchor_cz;

    float* axis_px;  float* axis_py;  float* axis_pz;

    float* limit_lo;
    float* limit_hi;

    float* target_pos;
    float* target_vel;
    float* stiffness;
    float* damping;

    float* compliance_pos;
    float* compliance_ang;

    uint8_t* type;

    uint32_t n;
};

} // namespace dyphur
