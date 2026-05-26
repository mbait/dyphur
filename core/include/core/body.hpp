#pragma once
#include "math/math.hpp"
#include <cstdint>

namespace dyphur {

// Body state bitmask flags.
namespace BodyFlag {
    constexpr uint32_t Static    = 1u << 0;  // infinite mass; never integrated
    constexpr uint32_t Kinematic = 1u << 1;  // velocity set externally; solver treats as infinite mass
    constexpr uint32_t Sleeping  = 1u << 2;  // island at rest; skip integration and solver
}

// Host-facing description of a body's initial state.
// Passed to BodyStore::add(); not used in kernels.
struct BodyParams {
    Vec3f    position         = Vec3f::zero();
    Quatf    rotation         = Quatf::identity();
    Vec3f    linear_velocity  = Vec3f::zero();
    Vec3f    angular_velocity = Vec3f::zero();
    float    mass             = 1.f;
    Mat3f    inertia          = Mat3f::identity();  // local body frame
    uint32_t shape_handle     = 0;
    uint32_t flags            = 0;
};

// Kernel-facing SoA view: raw device pointers + active body count.
// No owning semantics. Trivially copyable; safe to capture in a kernel lambda.
// All pointer arrays are indexed [0, n).
struct BodyView {
    // World-space pose
    float* pos_x;  float* pos_y;  float* pos_z;
    float* rot_w;  float* rot_x;  float* rot_y;  float* rot_z;
    // World-space velocity (linear + angular)
    float* vel_x;  float* vel_y;  float* vel_z;
    float* ang_x;  float* ang_y;  float* ang_z;
    // Precomputed dynamics (body-local frame, invariant across frames)
    float* inv_mass;
    float* iI_xx;  float* iI_yy;  float* iI_zz;  // diagonal of body-frame inverse inertia
    float* iI_xy;  float* iI_xz;  float* iI_yz;  // off-diagonal (symmetric)
    // Shape and flags
    uint32_t* shape;
    uint32_t* flags;
    // Active body count
    uint32_t n;
};

} // namespace dyphur
