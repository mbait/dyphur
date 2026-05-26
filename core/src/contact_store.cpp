#include <core/contact_store.hpp>
#include <sycl/sycl.hpp>

namespace dyphur {

ContactStore::ContactStore(Stream& s, uint32_t capacity)
    : cap_(capacity)
    , d_body_a_(s, capacity), d_body_b_(s, capacity)
    , d_pos_x_(s, capacity), d_pos_y_(s, capacity), d_pos_z_(s, capacity)
    , d_norm_x_(s, capacity), d_norm_y_(s, capacity), d_norm_z_(s, capacity)
    , d_depth_(s, capacity)
    , d_count_(s, 1)
{}

void ContactStore::reset(Stream& s) {
    s.queue().memset(d_count_.data(), 0, sizeof(uint32_t));
}

ContactView ContactStore::view() noexcept {
    return {
        d_body_a_.data(), d_body_b_.data(),
        d_pos_x_.data(),  d_pos_y_.data(),  d_pos_z_.data(),
        d_norm_x_.data(), d_norm_y_.data(), d_norm_z_.data(),
        d_depth_.data(),
        d_count_.data(),
        cap_
    };
}

uint32_t ContactStore::download_count(Stream& s) const {
    uint32_t cnt;
    s.queue().memcpy(&cnt, d_count_.data(), sizeof(cnt)).wait();
    return cnt;
}

} // namespace dyphur
