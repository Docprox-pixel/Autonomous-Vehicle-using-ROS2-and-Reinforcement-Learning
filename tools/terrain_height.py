#!/usr/bin/env python3
"""Height of the world's terrain meshes (MONTAIN / MOUNTAIN_TUNNEL ...) above the ground plane.

  python3 tools/terrain_height.py worlds/village.wbt X Y [RADIUS]

Loads every DEF'd Solid whose name contains MOUNTAIN/MONTAIN, applies its translation, axis-angle
rotation and Transform scale, and reports the highest mesh surface inside RADIUS of (X, Y) on a
1 m grid (barycentric interpolation of the triangles). Used to keep new asphalt off the hills.
"""
import math
import re
import sys

import numpy as np


def _rot(x, y, z, a):
    n = math.sqrt(x * x + y * y + z * z)
    x, y, z = x / n, y / n, z / n
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def meshes(world):
    text = open(world).read()
    out = []
    for m in re.finditer(r'(?m)^DEF (M\w*TAIN\w*) Solid \{', text):
        seg = text[m.end():m.end() + 3_000_000]
        tr = [float(v) for v in re.search(r'(?m)^  translation (.+)$', seg).group(1).split()]
        ro = [float(v) for v in re.search(r'(?m)^  rotation (.+)$', seg).group(1).split()]
        sc = re.search(r'scale ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)', seg)
        sc = np.array([float(v) for v in sc.groups()]) if sc else np.ones(3)
        pts = np.array(re.search(r'point \[\s*(.*?)\]', seg, re.S).group(1).replace(',', ' ').split(), float).reshape(-1, 3)
        idx = np.array(re.search(r'coordIndex \[\s*(.*?)\]', seg, re.S).group(1).replace(',', ' ').split(), int)
        faces = idx.reshape(-1, 4)[:, :3] if (idx.reshape(-1, 4)[:, 3] == -1).all() else None
        W = (pts * sc) @ _rot(*ro).T + tr
        out.append((m.group(1), W, faces))
    return out


def height_grid(world, cx, cy, R, step=1.0):
    best = {}
    for name, W, F in meshes(world):
        T = W[F]
        lo, hi = T[:, :, :2].min(1), T[:, :, :2].max(1)
        sel = (hi[:, 0] > cx - R) & (lo[:, 0] < cx + R) & (hi[:, 1] > cy - R) & (lo[:, 1] < cy + R)
        for tri in T[sel]:
            a, b, c = tri
            xs = np.arange(math.floor(min(tri[:, 0])), math.ceil(max(tri[:, 0])) + step, step)
            ys = np.arange(math.floor(min(tri[:, 1])), math.ceil(max(tri[:, 1])) + step, step)
            for x in xs:
                for y in ys:
                    if (x - cx) ** 2 + (y - cy) ** 2 > R * R:
                        continue
                    v0, v1, v2 = b[:2] - a[:2], c[:2] - a[:2], np.array([x, y]) - a[:2]
                    d = v0[0] * v1[1] - v1[0] * v0[1]
                    if abs(d) < 1e-9:
                        continue
                    u = (v2[0] * v1[1] - v1[0] * v2[1]) / d
                    w = (v0[0] * v2[1] - v2[0] * v0[1]) / d
                    if u >= -1e-6 and w >= -1e-6 and u + w <= 1 + 1e-6:
                        z = a[2] + u * (b[2] - a[2]) + w * (c[2] - a[2])
                        best[(x, y)] = max(best.get((x, y), -1e9), z)
    return best


if __name__ == '__main__':
    w, x, y = sys.argv[1], float(sys.argv[2]), float(sys.argv[3])
    R = float(sys.argv[4]) if len(sys.argv) > 4 else 15.0
    g = height_grid(w, x, y, R)
    up = {k: v for k, v in g.items() if v > 0.0}
    print(f'{len(g)} grid points with terrain within {R} m, {len(up)} above ground (max {max(g.values(), default=0):.2f} m)')
    for (px, py), z in sorted(up.items(), key=lambda kv: -kv[1])[:8]:
        print(f'  ({px:.0f},{py:.0f}) z {z:.2f}  dist {math.hypot(px - x, py - y):.1f}')
