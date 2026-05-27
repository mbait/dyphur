#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include <core/aabb.hpp>
#include <core/body_store.hpp>
#include <core/broadphase.hpp>
#include <core/contact_sensor.hpp>
#include <core/narrowphase.hpp>
#include <core/shape_store.hpp>
#include <compute/device.hpp>

namespace nb = nanobind;
using namespace nb::literals;
using namespace dyphur;

// Helper: zero-copy numpy view of a const uint32_t host array.
static auto uint32_view(const uint32_t* ptr, size_t n) {
    return nb::ndarray<nb::numpy, const uint32_t, nb::ndim<1>>(
        const_cast<uint32_t*>(ptr), {n}, nb::handle());
}

// Helper: zero-copy numpy view of a const float host array.
static auto float_view(const float* ptr, size_t n) {
    return nb::ndarray<nb::numpy, const float, nb::ndim<1>>(
        const_cast<float*>(ptr), {n}, nb::handle());
}

NB_MODULE(dyphur_py, m) {
    m.doc() = "dyphur GPU-first rigid-body physics (Python bindings)";

    // ── Device / Stream ───────────────────────────────────────────────────────

    nb::class_<Device>(m, "Device")
        .def_static("default_cpu", &Device::default_cpu)
        .def("make_stream", &Device::make_stream);

    nb::class_<Stream>(m, "Stream")
        .def("wait", &Stream::wait);

    // ── AABB ──────────────────────────────────────────────────────────────────

    nb::class_<AABB>(m, "AABB")
        .def(nb::init<float, float, float, float, float, float>(),
             "min_x"_a, "min_y"_a, "min_z"_a,
             "max_x"_a, "max_y"_a, "max_z"_a)
        .def_rw("min_x", &AABB::min_x)
        .def_rw("min_y", &AABB::min_y)
        .def_rw("min_z", &AABB::min_z)
        .def_rw("max_x", &AABB::max_x)
        .def_rw("max_y", &AABB::max_y)
        .def_rw("max_z", &AABB::max_z);

    // ── Shape ─────────────────────────────────────────────────────────────────

    nb::enum_<ShapeType>(m, "ShapeType")
        .value("Box",    ShapeType::Box)
        .value("Sphere", ShapeType::Sphere)
        .export_values();

    nb::class_<ShapeParams>(m, "ShapeParams")
        .def(nb::init<>())
        .def_rw("type",   &ShapeParams::type)
        .def_rw("half_x", &ShapeParams::half_x)
        .def_rw("half_y", &ShapeParams::half_y)
        .def_rw("half_z", &ShapeParams::half_z);

    nb::class_<ShapeStore>(m, "ShapeStore")
        .def(nb::init<Stream&, uint32_t>(), "s"_a, "capacity"_a)
        .def("add",      &ShapeStore::add)
        .def("upload",   &ShapeStore::upload)
        .def("count",    &ShapeStore::count)
        .def("view",     &ShapeStore::view, nb::rv_policy::reference_internal);

    nb::class_<ShapeView>(m, "ShapeView");

    // ── Body ──────────────────────────────────────────────────────────────────

    nb::class_<BodyParams>(m, "BodyParams")
        .def(nb::init<>())
        .def("set_position", [](BodyParams& p, float x, float y, float z) {
            p.position = Vec3f{x, y, z};
        }, "x"_a, "y"_a, "z"_a)
        .def("set_velocity", [](BodyParams& p, float vx, float vy, float vz) {
            p.linear_velocity = Vec3f{vx, vy, vz};
        }, "vx"_a, "vy"_a, "vz"_a)
        .def_rw("mass",         &BodyParams::mass)
        .def_rw("shape_handle", &BodyParams::shape_handle)
        .def_rw("flags",        &BodyParams::flags);

    nb::class_<BodyView>(m, "BodyView");

    nb::class_<BodyStore>(m, "BodyStore")
        .def(nb::init<Stream&, uint32_t>(), "s"_a, "capacity"_a)
        .def("add",          &BodyStore::add)
        .def("upload",       &BodyStore::upload)
        .def("count",        &BodyStore::count)
        .def("view",         &BodyStore::view, nb::rv_policy::reference_internal)
        .def("body_shapes",  &BodyStore::body_shapes, nb::rv_policy::reference_internal);

    // ── Broadphase ────────────────────────────────────────────────────────────

    nb::class_<Broadphase>(m, "Broadphase")
        .def(nb::init<Stream&, uint32_t, uint32_t>(),
             "s"_a, "max_bodies"_a, "max_pairs"_a)
        .def("build_and_query", &Broadphase::build_and_query,
             "s"_a, "bodies"_a, "shapes"_a, "scene_bounds"_a)
        .def("download_count", &Broadphase::download_count);

    // ── Narrowphase ───────────────────────────────────────────────────────────

    nb::class_<Narrowphase>(m, "Narrowphase")
        .def(nb::init<Stream&, uint32_t>(), "s"_a, "max_contacts"_a)
        .def("run", [](Narrowphase& np, Stream& s,
                       const Broadphase& bp, uint32_t n_pairs,
                       BodyView bv, ShapeView sv) {
            np.run(s, bp.pairs_ptr(), n_pairs, bv, sv);
        }, "s"_a, "bp"_a, "n_pairs"_a, "bodies"_a, "shapes"_a)
        .def("download_count", &Narrowphase::download_count);

    // ── ContactSensor ─────────────────────────────────────────────────────────

    nb::class_<ContactSensor>(m, "ContactSensor")
        .def(nb::init<uint32_t>(), "shape_idx"_a,
             "Subscribe to contacts involving the given shape index.")
        .def("query", &ContactSensor::query,
             "s"_a, "np"_a, "body_shapes"_a,
             "Download and filter contacts from the narrowphase. Returns contact count.")
        .def("count",     &ContactSensor::count)
        .def("shape_idx", &ContactSensor::shape_idx)
        // Zero-copy numpy views — reference_internal keeps ContactSensor alive.
        .def("body_a", [](const ContactSensor& cs) {
            return uint32_view(cs.body_a(), cs.count());
        }, nb::rv_policy::reference_internal,
           "Body-A indices (numpy uint32, zero-copy view).")
        .def("body_b", [](const ContactSensor& cs) {
            return uint32_view(cs.body_b(), cs.count());
        }, nb::rv_policy::reference_internal,
           "Body-B indices (numpy uint32, zero-copy view).")
        .def("pos_x", [](const ContactSensor& cs) {
            return float_view(cs.pos_x(), cs.count());
        }, nb::rv_policy::reference_internal,
           "Contact point X coordinates (numpy float32, zero-copy view).")
        .def("pos_y", [](const ContactSensor& cs) {
            return float_view(cs.pos_y(), cs.count());
        }, nb::rv_policy::reference_internal)
        .def("pos_z", [](const ContactSensor& cs) {
            return float_view(cs.pos_z(), cs.count());
        }, nb::rv_policy::reference_internal)
        .def("norm_x", [](const ContactSensor& cs) {
            return float_view(cs.norm_x(), cs.count());
        }, nb::rv_policy::reference_internal)
        .def("norm_y", [](const ContactSensor& cs) {
            return float_view(cs.norm_y(), cs.count());
        }, nb::rv_policy::reference_internal)
        .def("norm_z", [](const ContactSensor& cs) {
            return float_view(cs.norm_z(), cs.count());
        }, nb::rv_policy::reference_internal)
        .def("depth", [](const ContactSensor& cs) {
            return float_view(cs.depth(), cs.count());
        }, nb::rv_policy::reference_internal,
           "Penetration depths (numpy float32, zero-copy view).");
}
