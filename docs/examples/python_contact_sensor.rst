Python Contact Sensor
=====================

**Source**: ``bindings/python/tests/test_contact_sensor.py``

The Python module ``dyphur_py`` (built by ``cmake --build ... --target dyphur_py``)
exposes the full simulation setup API and the :class:`ContactSensor` for per-shape
contact event subscription.

Minimal example
---------------

.. code-block:: python

   import dyphur_py as dp
   import numpy as np

   # 1. Device + stream
   dev = dp.Device.default_cpu()
   s = dev.make_stream()

   # 2. Shapes
   ss = dp.ShapeStore(s, 4)
   sp = dp.ShapeParams()
   sp.type = dp.ShapeType.Sphere
   sp.half_x = 1.0        # radius = 1 m
   ss.add(sp)
   ss.upload()

   # 3. Bodies — two overlapping spheres
   bs = dp.BodyStore(s, 4)
   p = dp.BodyParams()
   p.mass = 1.0
   p.shape_handle = 0
   p.set_position(0.0, 0.0, 0.0); bs.add(p)
   p.set_position(1.5, 0.0, 0.0); bs.add(p)   # overlap = 0.5 m
   bs.upload()
   s.wait()

   # 4. Broadphase + narrowphase
   scene = dp.AABB(-2, -2, -2, 4, 2, 2)
   bp = dp.Broadphase(s, 4, 32)
   bp.build_and_query(s, bs.view(), ss.view(), scene)
   s.wait()

   n_pairs = bp.download_count(s)
   np_ = dp.Narrowphase(s, 32)
   np_.run(s, bp, n_pairs, bs.view(), ss.view())
   s.wait()

   # 5. Contact sensor for shape 0 (sphere)
   sensor = dp.ContactSensor(0)
   n = sensor.query(s, np_, bs.body_shapes())
   print(f"contacts: {n}")                  # → 1

   # Zero-copy numpy views (no data copy — OWNDATA is False)
   depth = np.asarray(sensor.depth())       # shape (1,)
   print(f"penetration depth: {depth[0]:.4f} m")  # → 0.5000 m

   pos_x = np.asarray(sensor.pos_x())      # contact point x
   norm_x = np.asarray(sensor.norm_x())    # contact normal x
   print(f"contact point x={pos_x[0]:.3f}, normal x={norm_x[0]:.3f}")

Zero-copy semantics
-------------------

Arrays returned by ``sensor.body_a()``, ``sensor.depth()``, ``sensor.pos_x()``, etc.
are **zero-copy views** into the sensor's host-side storage.  The underlying
``ContactSensor`` object is kept alive via nanobind's ``reference_internal``
policy while any view is alive.

.. code-block:: python

   arr = np.asarray(sensor.depth())
   assert not arr.flags["OWNDATA"]   # confirms no copy was made
