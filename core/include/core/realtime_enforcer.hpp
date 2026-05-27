#pragma once
#include <chrono>
#include <thread>

namespace dyphur {

// Adaptive timestep controller for real-time simulation.
//
// Call begin_frame() to get the dt to pass to the integrator.
// Call end_frame() after the GPU sync for that frame.
//
// The dt adapts to actual wallclock elapsed time so simulation time
// tracks real time. dt is always clamped to [dt_min, dt_max]:
//   - dt_min prevents physics from stalling when the sim runs very fast.
//   - dt_max prevents instability when the sim runs very slow.
//
// When limit_realtime is true, end_frame() sleeps until dt_nominal has
// elapsed, so the simulation never runs faster than realtime.

struct RealtimeEnforcerParams {
    float dt_nominal     = 1.f / 60.f;   // target step (s)
    float dt_min         = 1.f / 240.f;  // minimum step (s) — caps max speed
    float dt_max         = 1.f / 15.f;   // maximum step (s) — caps slowdown
    bool  limit_realtime = false;         // sleep to stay at <= realtime pace
};

class RealtimeEnforcer {
public:
    using Clock   = std::chrono::steady_clock;
    using Seconds = std::chrono::duration<double>;
    using Params  = RealtimeEnforcerParams;

    explicit RealtimeEnforcer(Params p = Params{})
        : p_(p), dt_(p.dt_nominal) {}

    // Returns the dt to use for this simulation frame.
    // On the first call returns dt_nominal; subsequent calls return the
    // clamped elapsed time since the previous begin_frame().
    float begin_frame() {
        auto now = Clock::now();
        if (started_) {
            double wall = Seconds(now - frame_start_).count();
            double clamped = wall;
            if (clamped < p_.dt_min) clamped = p_.dt_min;
            if (clamped > p_.dt_max) clamped = p_.dt_max;
            dt_ = static_cast<float>(clamped);
        }
        frame_start_ = now;
        started_     = true;
        return dt_;
    }

    // Call after the GPU sync for this frame.
    // Records frame time for the next begin_frame(); sleeps if limit_realtime.
    // Returns actual wall time (seconds) this frame took.
    float end_frame() {
        auto now  = Clock::now();
        double elapsed = Seconds(now - frame_start_).count();

        if (p_.limit_realtime && elapsed < p_.dt_nominal) {
            auto sleep_dur = std::chrono::duration<double>(p_.dt_nominal - elapsed);
            std::this_thread::sleep_for(sleep_dur);
            elapsed = Seconds(Clock::now() - frame_start_).count();
        }

        total_sim_time_ += dt_;
        total_wall_time_ = Seconds(now - sim_start_).count();
        rt_factor_ = (total_wall_time_ > 0.0)
                     ? static_cast<float>(total_sim_time_ / total_wall_time_)
                     : 1.f;
        return static_cast<float>(elapsed);
    }

    // Current adapted timestep (set by the most recent begin_frame()).
    float dt()             const { return dt_; }

    // Ratio of accumulated simulation time to elapsed wall time.
    // > 1: faster than realtime; < 1: slower than realtime.
    float realtime_factor() const { return rt_factor_; }

    bool is_realtime()      const { return rt_factor_ >= 0.95f; }

private:
    Params  p_;
    float   dt_              = 0.f;
    float   rt_factor_       = 1.f;
    double  total_sim_time_  = 0.0;
    double  total_wall_time_ = 0.0;
    bool    started_         = false;
    Clock::time_point frame_start_  = Clock::now();
    Clock::time_point sim_start_    = Clock::now();
};

} // namespace dyphur
