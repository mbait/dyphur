#pragma once
#include <sycl/sycl.hpp>

namespace dyphur {

class Event {
public:
    Event() = default;
    explicit Event(sycl::event e) : e_(std::move(e)) {}
    void wait() { e_.wait_and_throw(); }
    const sycl::event& sycl_event() const { return e_; }
private:
    sycl::event e_;
};

// Wraps a SYCL in-order queue. Successive submissions execute in order;
// no explicit inter-kernel barriers are needed.
class Stream {
public:
    explicit Stream(const sycl::device& dev)
        : queue_(dev, sycl::property::queue::in_order{}) {}

    sycl::queue& queue()             { return queue_; }
    const sycl::queue& queue() const { return queue_; }

    void wait() { queue_.wait_and_throw(); }

private:
    sycl::queue queue_;
};

} // namespace dyphur
