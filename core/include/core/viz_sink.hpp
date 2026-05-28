#pragma once
// VizSink: optional live-mode producer.
//
// The simulation can push body poses to the live visualiser over a POSIX
// shared-memory ring buffer.  If no visualiser is attached, push() is a no-op.
// The viz process does not need to be running for the simulation to compile or run.
//
// The ring buffer is a single shared frame — producer overwrites, consumer reads
// the latest.  Thread-safety: one writer (simulation), one reader (viz process).
//
// Usage:
//   VizSink sink("my_sim");          // open/create shared segment
//   sink.push(bv, n);                // call each frame after solver
//   // VizSink destructor closes the segment

#include <core/body.hpp>
#include <cstdint>
#include <cstring>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

namespace dyphur {

class VizSink {
public:
    // Open or create the shared segment named "/dyphur_viz_<name>".
    // max_bodies: upper bound on body count (determines segment size).
    explicit VizSink(const char* name, uint32_t max_bodies = 4096)
        : _max_bodies{max_bodies}
    {
        // Segment layout: Header (16 bytes) + max_bodies * 7 floats.
        _seg_size = sizeof(Header) + max_bodies * 7 * sizeof(float);
        char seg_name[64];
        std::snprintf(seg_name, sizeof(seg_name), "/dyphur_viz_%s", name);

        int fd = ::shm_open(seg_name, O_CREAT | O_RDWR, 0666);
        if (fd < 0) return;   // silently disabled if shm_open fails
        if (::ftruncate(fd, static_cast<off_t>(_seg_size)) < 0) {
            ::close(fd);
            return;
        }
        _seg = static_cast<char*>(
            ::mmap(nullptr, _seg_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0));
        ::close(fd);
        if (_seg == MAP_FAILED) { _seg = nullptr; return; }

        auto* h = header();
        h->magic      = 0x44595048u; // 'DYPH'
        h->max_bodies = max_bodies;
        h->n_bodies   = 0;
        h->seq        = 0;
    }

    ~VizSink() {
        if (_seg) ::munmap(_seg, _seg_size);
    }

    VizSink(const VizSink&) = delete;
    VizSink& operator=(const VizSink&) = delete;

    bool attached() const { return _seg != nullptr; }

    // Push the current body poses.  n = number of bodies to copy.
    // Reads pos_x/y/z and rot_w/x/y/z directly from host-side device buffer
    // pointers (all SoA).  Must be called after stream.wait() so data is visible.
    void push(const BodyView& bv, uint32_t n) {
        if (!_seg || n > _max_bodies) return;
        auto* h = header();
        h->n_bodies = n;
        float* dst = poses();
        for (uint32_t i = 0; i < n; ++i) {
            float* p = dst + i * 7;
            p[0] = bv.pos_x[i]; p[1] = bv.pos_y[i]; p[2] = bv.pos_z[i];
            p[3] = bv.rot_w[i]; p[4] = bv.rot_x[i]; p[5] = bv.rot_y[i]; p[6] = bv.rot_z[i];
        }
        // Increment sequence number last so the consumer sees a consistent frame.
        __atomic_fetch_add(&h->seq, 1u, __ATOMIC_RELEASE);
    }

private:
    struct Header {
        uint32_t magic;
        uint32_t max_bodies;
        uint32_t n_bodies;
        uint32_t seq;
    };

    Header* header() { return reinterpret_cast<Header*>(_seg); }
    float*  poses()  { return reinterpret_cast<float*>(_seg + sizeof(Header)); }

    char*    _seg      = nullptr;
    size_t   _seg_size = 0;
    uint32_t _max_bodies;
};

} // namespace dyphur
