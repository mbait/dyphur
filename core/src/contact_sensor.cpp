#include <core/contact_sensor.hpp>
#include <algorithm>

namespace dyphur {

uint32_t ContactSensor::query(Stream& s,
                              const Narrowphase& np,
                              const std::vector<uint32_t>& body_shapes) {
    // Download live contact count (blocking).
    uint32_t total = const_cast<Narrowphase&>(np).download_count(s);

    count_ = 0;
    h_body_a_.clear(); h_body_b_.clear();
    h_pos_x_.clear();  h_pos_y_.clear();  h_pos_z_.clear();
    h_norm_x_.clear(); h_norm_y_.clear(); h_norm_z_.clear();
    h_depth_.clear();

    if (total == 0) return 0;

    // Download all contact arrays (blocking memcpy per field).
    auto cv = const_cast<Narrowphase&>(np).contacts();
    auto& q = s.queue();

    std::vector<uint32_t> ba(total), bb(total);
    std::vector<float>    px(total), py(total), pz(total);
    std::vector<float>    nx(total), ny(total), nz(total);
    std::vector<float>    dep(total);

    q.memcpy(ba.data(),  cv.body_a, total * sizeof(uint32_t));
    q.memcpy(bb.data(),  cv.body_b, total * sizeof(uint32_t));
    q.memcpy(px.data(),  cv.pos_x,  total * sizeof(float));
    q.memcpy(py.data(),  cv.pos_y,  total * sizeof(float));
    q.memcpy(pz.data(),  cv.pos_z,  total * sizeof(float));
    q.memcpy(nx.data(),  cv.norm_x, total * sizeof(float));
    q.memcpy(ny.data(),  cv.norm_y, total * sizeof(float));
    q.memcpy(nz.data(),  cv.norm_z, total * sizeof(float));
    q.memcpy(dep.data(), cv.depth,  total * sizeof(float));
    s.wait();

    // Filter to contacts where body_a or body_b has shape shape_idx_.
    for (uint32_t i = 0; i < total; ++i) {
        uint32_t a = ba[i], b = bb[i];
        bool a_match = (a < body_shapes.size()) && (body_shapes[a] == shape_idx_);
        bool b_match = (b < body_shapes.size()) && (body_shapes[b] == shape_idx_);
        if (!a_match && !b_match) continue;

        h_body_a_.push_back(a);
        h_body_b_.push_back(b);
        h_pos_x_.push_back(px[i]);
        h_pos_y_.push_back(py[i]);
        h_pos_z_.push_back(pz[i]);
        h_norm_x_.push_back(nx[i]);
        h_norm_y_.push_back(ny[i]);
        h_norm_z_.push_back(nz[i]);
        h_depth_.push_back(dep[i]);
    }
    count_ = static_cast<uint32_t>(h_depth_.size());
    return count_;
}

} // namespace dyphur
