Robotic Arm Push (C++)
======================

**Source**: ``examples/arm_push/main.cpp``

Overview
--------

Simulates a 6-DoF robotic arm (modelled as a kinematic chain of boxes and spheres)
pushing a pile of rigid blocks off a table.  The simulation runs headless for 600
frames and writes:

* ``arm_push.trajectory`` — body-state dump (position + quaternion every 2 frames)
* ``arm_push.metrics.json`` — fps, contact counts, determinism hash

Running
-------

.. code-block:: bash

   cmake --preset=omp
   cmake --build build/omp --config Release --target arm_push
   build/omp/examples/arm_push/Release/arm_push

Expected output::

   frames=600  fps=6579  contacts_avg=47  rt_factor=109.6x  hash=0x1a2b3c4d

Physics pipeline per frame
--------------------------

.. code-block:: text

   begin_frame() → dt
       ├─ integrate(dt)           velocity + position update (symplectic Euler)
       ├─ broadphase.build_and_query()  LBVH build + BVH traversal for pairs
       ├─ broadphase.sort_pairs()       deterministic pair order
       ├─ narrowphase.run()             per-pair exact contact detection
       └─ solver.solve()               XPBD constraint resolution
   end_frame()

Key classes
-----------

* :class:`dyphur::BodyStore` — hosts rigid body SoA arrays, uploads to device
* :class:`dyphur::Broadphase` — LBVH-based broad collision detection
* :class:`dyphur::Narrowphase` — exact sphere/box/convex-hull contact detection
* :class:`dyphur::RealtimeEnforcer` — adaptive timestep controller
