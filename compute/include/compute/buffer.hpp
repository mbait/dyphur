#pragma once
#include "stream.hpp"
#include <sycl/sycl.hpp>
#include <cstddef>

namespace dyphur {

// Device-resident buffer backed by USM device memory.
// Upload/download transfers go through the Stream's queue.
template<typename T>
class Buffer {
public:
    Buffer() = default;

    Buffer(Stream& s, size_t n)
        : q_(&s.queue()), n_(n),
          data_(n > 0 ? sycl::malloc_device<T>(n, s.queue()) : nullptr) {}

    ~Buffer() { if (data_) sycl::free(data_, *q_); }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& o) noexcept : q_(o.q_), n_(o.n_), data_(o.data_) {
        o.data_ = nullptr; o.n_ = 0;
    }
    Buffer& operator=(Buffer&& o) noexcept {
        if (this != &o) {
            if (data_) sycl::free(data_, *q_);
            q_ = o.q_; n_ = o.n_; data_ = o.data_;
            o.data_ = nullptr; o.n_ = 0;
        }
        return *this;
    }

    T*       data()       noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    size_t   size() const noexcept { return n_; }
    bool     empty()const noexcept { return n_ == 0; }

    // Async upload from host; submit on the stream's queue.
    Event upload(const T* src, size_t count = 0) {
        return Event(q_->memcpy(data_, src, (count ? count : n_) * sizeof(T)));
    }

    // Blocking download to host.
    void download(T* dst, size_t count = 0) const {
        q_->memcpy(dst, data_, (count ? count : n_) * sizeof(T)).wait();
    }

private:
    sycl::queue* q_   = nullptr;
    size_t       n_   = 0;
    T*           data_= nullptr;
};

} // namespace dyphur
