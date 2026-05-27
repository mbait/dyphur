#include <core/convex_hull_store.hpp>
#include <cassert>

namespace dyphur {

ConvexHullStore::ConvexHullStore(Stream& s, uint32_t max_hulls, uint32_t max_total_verts)
    : max_hulls_(max_hulls), max_verts_(max_total_verts)
    , d_vtx_x_(s, max_total_verts), d_vtx_y_(s, max_total_verts), d_vtx_z_(s, max_total_verts)
    , d_hull_start_(s, max_hulls), d_hull_count_(s, max_hulls)
{
    h_vtx_x_.reserve(max_total_verts);
    h_vtx_y_.reserve(max_total_verts);
    h_vtx_z_.reserve(max_total_verts);
    h_hull_start_.reserve(max_hulls);
    h_hull_count_.reserve(max_hulls);
}

uint32_t ConvexHullStore::add(const float* x, const float* y, const float* z, uint32_t n) {
    assert(n_hulls_ < max_hulls_ && "ConvexHullStore hull capacity exceeded");
    assert(n_verts_ + n <= max_verts_ && "ConvexHullStore vertex capacity exceeded");
    h_hull_start_.push_back(n_verts_);
    h_hull_count_.push_back(n);
    for (uint32_t i = 0; i < n; ++i) {
        h_vtx_x_.push_back(x[i]);
        h_vtx_y_.push_back(y[i]);
        h_vtx_z_.push_back(z[i]);
    }
    n_verts_ += n;
    return n_hulls_++;
}

void ConvexHullStore::upload() {
    if (n_hulls_ == 0) return;
    d_vtx_x_.upload(h_vtx_x_.data(), n_verts_);
    d_vtx_y_.upload(h_vtx_y_.data(), n_verts_);
    d_vtx_z_.upload(h_vtx_z_.data(), n_verts_);
    d_hull_start_.upload(h_hull_start_.data(), n_hulls_);
    d_hull_count_.upload(h_hull_count_.data(), n_hulls_);
}

ConvexHullView ConvexHullStore::view() const noexcept {
    return { d_vtx_x_.data(), d_vtx_y_.data(), d_vtx_z_.data(),
             d_hull_start_.data(), d_hull_count_.data(),
             n_hulls_, n_verts_ };
}

} // namespace dyphur
