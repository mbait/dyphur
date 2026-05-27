"""
Contact-sensor smoke tests.

Two touching spheres (radius 1, centres at 0 and 1.5 → overlap 0.5 m).
Shape 0 is the sphere shape.  ContactSensor(0) should report exactly 1 contact
after one narrowphase run.  All numpy arrays must be zero-copy (no copy flags).
"""
import numpy as np
import pytest
import dyphur_py as dp


@pytest.fixture(scope="module")
def two_sphere_scene():
    """Returns (stream, bp, np_, bs, ss, n_pairs) for a two-touching-sphere scene."""
    dev = dp.Device.default_cpu()
    s = dev.make_stream()

    # Shape catalogue: one sphere with radius 1.
    ss = dp.ShapeStore(s, 4)
    sp = dp.ShapeParams()
    sp.type = dp.ShapeType.Sphere
    sp.half_x = 1.0
    ss.add(sp)
    ss.upload()

    # Two bodies: sphere A at (0,0,0), sphere B at (1.5,0,0).
    # Distance between centres = 1.5 < r_A + r_B = 2  → overlap = 0.5 m.
    bs = dp.BodyStore(s, 4)
    p = dp.BodyParams()
    p.mass = 1.0
    p.shape_handle = 0
    p.set_position(0.0, 0.0, 0.0)
    bs.add(p)
    p.set_position(1.5, 0.0, 0.0)
    bs.add(p)
    bs.upload()
    s.wait()

    scene = dp.AABB(-2.0, -2.0, -2.0, 4.0, 2.0, 2.0)
    bp = dp.Broadphase(s, 4, 32)
    bp.build_and_query(s, bs.view(), ss.view(), scene)
    s.wait()

    n_pairs = bp.download_count(s)
    np_ = dp.Narrowphase(s, 32)
    np_.run(s, bp, n_pairs, bs.view(), ss.view())
    s.wait()

    return s, bp, np_, bs, ss, n_pairs


def test_broadphase_finds_pair(two_sphere_scene):
    _, _, _, _, _, n_pairs = two_sphere_scene
    assert n_pairs == 1


def test_narrowphase_finds_contact(two_sphere_scene):
    s, _, np_, bs, _, _ = two_sphere_scene
    n_contacts = np_.download_count(s)
    assert n_contacts == 1


def test_contact_sensor_count(two_sphere_scene):
    s, _, np_, bs, _, _ = two_sphere_scene
    sensor = dp.ContactSensor(0)
    cnt = sensor.query(s, np_, bs.body_shapes())
    assert cnt == 1
    assert sensor.count() == 1


def test_contact_sensor_depth_positive(two_sphere_scene):
    s, _, np_, bs, _, _ = two_sphere_scene
    sensor = dp.ContactSensor(0)
    sensor.query(s, np_, bs.body_shapes())
    depth = sensor.depth()
    assert len(depth) == 1
    assert float(depth[0]) > 0.0


def test_contact_sensor_numpy_zero_copy(two_sphere_scene):
    """Verify numpy arrays returned by the sensor are views (no data copy)."""
    s, _, np_, bs, _, _ = two_sphere_scene
    sensor = dp.ContactSensor(0)
    sensor.query(s, np_, bs.body_shapes())

    # Zero-copy: the array flags must NOT own data (OWNDATA = False).
    arr = np.asarray(sensor.depth())
    assert not arr.flags["OWNDATA"], "depth() should be a zero-copy view"

    arr_a = np.asarray(sensor.body_a())
    assert not arr_a.flags["OWNDATA"], "body_a() should be a zero-copy view"


def test_contact_sensor_normal_unit_length(two_sphere_scene):
    """Contact normal should be a unit vector."""
    s, _, np_, bs, _, _ = two_sphere_scene
    sensor = dp.ContactSensor(0)
    sensor.query(s, np_, bs.body_shapes())

    nx = float(sensor.norm_x()[0])
    ny = float(sensor.norm_y()[0])
    nz = float(sensor.norm_z()[0])
    mag = (nx**2 + ny**2 + nz**2) ** 0.5
    assert abs(mag - 1.0) < 1e-4


def test_contact_sensor_no_hit_wrong_shape(two_sphere_scene):
    """A sensor for shape index 99 (doesn't exist) should find 0 contacts."""
    s, _, np_, bs, _, _ = two_sphere_scene
    sensor = dp.ContactSensor(99)
    cnt = sensor.query(s, np_, bs.body_shapes())
    assert cnt == 0
