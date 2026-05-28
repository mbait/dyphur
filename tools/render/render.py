#!/usr/bin/env python3
"""
tools/render/render.py — Python isometric renderer for dyphur trajectory files.

Reads a <prefix>.scene + <prefix>.trajectory pair and produces PNG snapshots
and/or an optional settling-height time-series plot.  Useful as a quick
debugging alternative to the C++ viz tool — no GPU or display required.

Usage:
  render.py <prefix> [options]

Dependencies: numpy, Pillow, matplotlib (optional, for --settling)
Install in the project venv:
  python3 -m venv .venv && source .venv/bin/activate && pip install numpy pillow matplotlib
"""

import argparse
import math
import os
import struct
import sys

import numpy as np
from PIL import Image, ImageDraw


# ── Scene file reading ────────────────────────────────────────────────────────

SHAPE_BOX          = 0
SHAPE_SPHERE       = 1
SHAPE_CONVEX_HULL  = 2
SHAPE_TRIANGLE_MESH = 3


def read_scene(prefix):
    """Return (n_bodies, body_shape_idx, shapes) from <prefix>.scene.

    shapes is a list of dicts: {type, half_x, half_y, half_z}
    """
    path = prefix + ".scene"
    with open(path, "rb") as f:
        n_bodies, n_shapes = struct.unpack("<II", f.read(8))
        body_shape_idx = list(struct.unpack(f"<{n_bodies}I", f.read(n_bodies * 4)))
        shapes = []
        for _ in range(n_shapes):
            stype, hx, hy, hz, _pad = struct.unpack("<IfffI", f.read(20))
            shapes.append({"type": stype, "half_x": hx, "half_y": hy, "half_z": hz})
    return n_bodies, body_shape_idx, shapes


def read_trajectory(prefix, n_bodies):
    """Return ndarray[T, n_bodies, 7] (x,y,z, qw,qx,qy,qz) from <prefix>.trajectory."""
    path = prefix + ".trajectory"
    row_bytes = n_bodies * 7 * 4
    frames = []
    with open(path, "rb") as f:
        hdr_n, _hdr_reserved = struct.unpack("<II", f.read(8))
        if hdr_n != n_bodies:
            raise ValueError(
                f"{path}: header says {hdr_n} bodies but scene says {n_bodies}"
            )
        while True:
            raw = f.read(row_bytes)
            if len(raw) < row_bytes:
                break
            frames.append(np.frombuffer(raw, dtype="<f4").reshape(n_bodies, 7).copy())
    if not frames:
        raise ValueError(f"{path}: no frames found")
    return np.stack(frames, axis=0)  # [T, N, 7]


# ── Quaternion / geometry helpers ─────────────────────────────────────────────

def quat_rotate(v, qw, qx, qy, qz):
    q = np.array([qx, qy, qz], dtype=float)
    v = np.asarray(v, dtype=float)
    t = 2.0 * np.cross(q, v)
    return v + qw * t + np.cross(q, t)


def box_corners(cx, cy, cz, qw, qx, qy, qz, hx, hy, hz):
    """8 world-space corners of an oriented box."""
    c = np.array([cx, cy, cz])
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx * hx, sy * hy, sz * hz])
                corners.append(c + quat_rotate(local, qw, qx, qy, qz))
    return corners


# ── Isometric projection ──────────────────────────────────────────────────────
# Standard 2:1 cabinet: x→right+down, z→left+down, y→up.
# Camera view direction (world-space, unnormalised): (1, 2, -1).

def iso(x, y, z, ox, oy, scale):
    sx = (x - z) * scale + ox
    sy = -(x + z) * 0.5 * scale - y * scale + oy
    return (sx, sy)


def depth_key(p):
    return p[0] + p[2] - 2.0 * p[1]


# ── Drawing primitives ────────────────────────────────────────────────────────

_CAM_DIR = np.array([1.0, 2.0, -1.0])

_BOX_FACES = [
    ([2, 6, 7, 3], np.array([ 0.,  1.,  0.])),  # +Y top
    ([0, 4, 5, 1], np.array([ 0., -1.,  0.])),  # -Y bottom
    ([4, 6, 7, 5], np.array([ 1.,  0.,  0.])),  # +X right
    ([0, 2, 3, 1], np.array([-1.,  0.,  0.])),  # -X left
    ([1, 5, 7, 3], np.array([ 0.,  0.,  1.])),  # +Z front
    ([0, 4, 6, 2], np.array([ 0.,  0., -1.])),  # -Z back
]


def draw_oriented_box(draw, cx, cy, cz, qw, qx, qy, qz,
                      hx, hy, hz, color, ox, oy, scale):
    cn = box_corners(cx, cy, cz, qw, qx, qy, qz, hx, hy, hz)
    visible = []
    for idxs, local_norm in _BOX_FACES:
        world_norm = quat_rotate(local_norm, qw, qx, qy, qz)
        if np.dot(world_norm, _CAM_DIR) <= 0.0:
            continue
        pts = [cn[i] for i in idxs]
        d = depth_key(np.mean(pts, axis=0))
        visible.append((d, pts, local_norm))
    visible.sort(key=lambda x: -x[0])

    for _d, pts, local_norm in visible:
        dot_top   = abs(np.dot(local_norm, [0., 1., 0.]))
        dot_front = abs(np.dot(local_norm, [0., 0., 1.]))
        dot_right = abs(np.dot(local_norm, [1., 0., 0.]))
        shade = 1.0 * dot_top + 0.65 * dot_front + 0.55 * dot_right + \
                0.45 * (1.0 - dot_top - dot_front - dot_right)
        shade = max(0.4, min(1.0, shade))
        fill = tuple(int(c * shade) for c in color)
        screen = [iso(p[0], p[1], p[2], ox, oy, scale) for p in pts]
        draw.polygon(screen, fill=fill)
        draw.line(screen + [screen[0]], fill=(40, 40, 40), width=1)


def draw_sphere(draw, cx, cy, cz, radius, color, ox, oy, scale):
    # Project the top, bottom, left, right extents of the sphere into screen
    # space to get an ellipse approximation (good enough for debugging).
    cx_s, cy_s = iso(cx, cy, cz, ox, oy, scale)
    rx = radius * scale          # horizontal extent (x-z axis combined)
    ry = radius * scale * 0.5   # vertical extent (iso squash factor)
    bbox = [cx_s - rx, cy_s - ry, cx_s + rx, cy_s + ry]
    shade = 0.8
    fill = tuple(int(c * shade) for c in color)
    draw.ellipse(bbox, fill=fill, outline=(40, 40, 40))


# ── Colour scheme ─────────────────────────────────────────────────────────────

# Palette identical to the C++ Renderer for cross-tool consistency.
_PALETTE = [
    (0x44, 0x66, 0xcc),
    (0x44, 0xcc, 0x66),
    (0xcc, 0x66, 0x44),
    (0xcc, 0xaa, 0x44),
    (0x44, 0xcc, 0xcc),
    (0xcc, 0x44, 0xcc),
    (0xaa, 0xcc, 0x44),
    (0x44, 0x88, 0xff),
]


def body_color(i):
    return _PALETTE[i % len(_PALETTE)]


def height_color(y, y_min=0.0, y_max=18.0):
    """Map height to a blue→orange gradient, used for dense scenes."""
    t = max(0.0, min(1.0, (y - y_min) / (y_max - y_min)))
    if t < 0.33:
        s = t / 0.33
        return (int(20 + 60 * s), int(80 + 120 * s), int(200 - 40 * s))
    elif t < 0.66:
        s = (t - 0.33) / 0.33
        return (int(80 + 120 * s), int(200 - 20 * s), int(160 - 100 * s))
    else:
        s = (t - 0.66) / 0.34
        return (int(200 + 40 * s), int(180 - 120 * s), int(60 - 40 * s))


# ── Frame renderer ────────────────────────────────────────────────────────────

def render_frame(frame, n_bodies, body_shape_idx, shapes,
                 title="", width=1200, height=800, scale=20.0,
                 color_by_height=False):
    img = Image.new("RGB", (width, height), (34, 34, 34))
    draw = ImageDraw.Draw(img)

    ox = width * 0.55
    oy = height * 0.82

    # Collect all bodies with their depth for painter's-algorithm sort.
    body_depths = []
    for i in range(n_bodies):
        cx, cy, cz = frame[i, 0], frame[i, 1], frame[i, 2]
        body_depths.append((depth_key(np.array([cx, cy, cz])), i))
    body_depths.sort(key=lambda x: -x[0])

    for _, i in body_depths:
        cx, cy, cz = frame[i, 0], frame[i, 1], frame[i, 2]
        qw, qx, qy, qz = frame[i, 3], frame[i, 4], frame[i, 5], frame[i, 6]
        si = body_shape_idx[i]
        sp = shapes[si]
        hx, hy, hz = sp["half_x"], sp["half_y"], sp["half_z"]

        if color_by_height:
            color = height_color(cy)
        else:
            color = body_color(i)

        stype = sp["type"]
        if stype == SHAPE_SPHERE:
            draw_sphere(draw, cx, cy, cz, hx, color, ox, oy, scale)
        else:
            # Box, ConvexHull (bbox fallback), TriangleMesh (bbox fallback)
            if hx == 0 and hy == 0 and hz == 0:
                hx = hy = hz = 0.1
            draw_oriented_box(draw, cx, cy, cz, qw, qx, qy, qz,
                               hx, hy, hz, color, ox, oy, scale)

    if title:
        draw.text((20, 18), title, fill=(220, 220, 220))
    draw.text((20, 40),
              f"bodies: {n_bodies}  scale: {scale:.0f} px/m",
              fill=(160, 160, 160))

    return img


# ── Settling plot ─────────────────────────────────────────────────────────────

def render_settling(traj, prefix, out_dir, sim_dt=1.0 / 60.0):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib not available — skipping --settling plot", file=sys.stderr)
        return None

    T, N, _ = traj.shape
    avg_y = traj[:, :, 1].mean(axis=1)
    min_y = traj[:, :, 1].min(axis=1)
    max_y = traj[:, :, 1].max(axis=1)
    t_axis = np.arange(T) * sim_dt

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#222222")
    ax.set_facecolor("#2a2a2a")
    ax.fill_between(t_axis, min_y, max_y, alpha=0.25, color="#4488ff", label="min–max range")
    ax.plot(t_axis, avg_y, color="#4488ff", lw=2, label="mean height")
    ax.set_xlabel("Simulation time (s)", color="#cccccc")
    ax.set_ylabel("Body height y (m)", color="#cccccc")
    ax.set_title(f"{os.path.basename(prefix)} — body height over time", color="#eeeeee")
    ax.tick_params(colors="#aaaaaa")
    ax.spines[:].set_color("#555555")
    ax.legend(framealpha=0.2, labelcolor="white")

    out = os.path.join(out_dir, os.path.basename(prefix) + "_settling.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Isometric PNG renderer for dyphur trajectory files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("prefix",
                   help="Path prefix — reads <prefix>.scene and <prefix>.trajectory")
    p.add_argument("--frames", default=None,
                   help="Comma-separated frame indices or 'first,mid,last' (default: first,mid,last)")
    p.add_argument("--output-dir", "-o", default=".",
                   help="Directory for output PNGs (default: current directory)")
    p.add_argument("--width",  type=int, default=1200)
    p.add_argument("--height", type=int, default=800)
    p.add_argument("--scale",  type=float, default=20.0,
                   help="Pixels per metre (default: 20)")
    p.add_argument("--color-by-height", action="store_true",
                   help="Use height-gradient colouring instead of body-index palette")
    p.add_argument("--settling", action="store_true",
                   help="Also emit a <prefix>_settling.png height time-series plot")
    p.add_argument("--sim-dt", type=float, default=1.0 / 60.0,
                   help="Simulation timestep in seconds (for settling plot x-axis)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Reading scene: {args.prefix}.scene")
    n_bodies, body_shape_idx, shapes = read_scene(args.prefix)
    print(f"  {n_bodies} bodies, {len(shapes)} shape(s)")

    print(f"Reading trajectory: {args.prefix}.trajectory")
    traj = read_trajectory(args.prefix, n_bodies)
    T = traj.shape[0]
    print(f"  {T} frames")

    if args.frames is None:
        frame_indices = sorted({0, T // 2, T - 1})
    else:
        frame_indices = []
        for tok in args.frames.split(","):
            tok = tok.strip()
            if tok == "first":
                frame_indices.append(0)
            elif tok == "mid":
                frame_indices.append(T // 2)
            elif tok == "last":
                frame_indices.append(T - 1)
            else:
                frame_indices.append(int(tok))

    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.basename(args.prefix)

    for fi in frame_indices:
        if fi < 0 or fi >= T:
            print(f"warning: frame {fi} out of range [0, {T-1}], skipping", file=sys.stderr)
            continue
        title = f"{name} — frame {fi}/{T-1}"
        img = render_frame(
            traj[fi], n_bodies, body_shape_idx, shapes,
            title=title,
            width=args.width, height=args.height, scale=args.scale,
            color_by_height=args.color_by_height,
        )
        out = os.path.join(args.output_dir, f"{name}_f{fi:04d}.png")
        img.save(out)
        print(f"  wrote {out}")

    if args.settling:
        out = render_settling(traj, args.prefix, args.output_dir, args.sim_dt)
        if out:
            print(f"  wrote {out}")


if __name__ == "__main__":
    main()
