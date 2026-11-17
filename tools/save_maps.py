#!/usr/bin/env python3
"""Save the live maps as images (no RViz needed).

  python3 tools/save_maps.py OUT_DIR
    OUT_DIR/slam_map.png       2D SLAM map (/map)              white free, black obstacle, grey unknown
    OUT_DIR/projected_map.png  2D map projected from the 3D map (/projected_map)
    OUT_DIR/octomap_3d.png     3D voxel map (/occupied_cells_vis_array), oblique view, colour = height
"""
import os
import sys
import time

import numpy as np
import rclpy
from nav_msgs.msg import OccupancyGrid
from PIL import Image, ImageDraw
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import MarkerArray


def grid_png(m, path):
    g = np.asarray(m.data, dtype=np.int16).reshape(m.info.height, m.info.width)
    img = np.full(g.shape, 205, np.uint8)
    img[g == 0] = 255
    img[g > 50] = 0
    img = np.flipud(img)
    known = np.where(img != 205)
    if len(known[0]):
        pad = 10
        img = img[max(known[0].min() - pad, 0):known[0].max() + pad, max(known[1].min() - pad, 0):known[1].max() + pad]
    scale = max(1, int(900 / max(img.shape)))
    Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST).save(path)
    return int((g > 50).sum()), int((g == 0).sum()) * m.info.resolution ** 2


def octomap_png(markers, path):
    pts = []
    for mk in markers.markers:
        pts += [(p.x, p.y, p.z) for p in mk.points]
    if not pts:
        return 0
    p = np.asarray(pts)
    # oblique view: rotate 35 deg around z, tilt 55 deg
    a, t = np.radians(35), np.radians(55)
    x = p[:, 0] * np.cos(a) - p[:, 1] * np.sin(a)
    y = p[:, 0] * np.sin(a) + p[:, 1] * np.cos(a)
    u, v, depth = x, y * np.cos(t) - p[:, 2] * np.sin(t), y * np.sin(t) + p[:, 2] * np.cos(t)
    W = 1400
    s = (W - 40) / max(np.ptp(u), np.ptp(v), 1e-6)
    U = ((u - u.min()) * s + 20).astype(int)
    V = ((v.max() - v) * s + 20).astype(int)
    H = int(V.max() + 20)
    img = Image.new('RGB', (W, H), (25, 25, 30))
    d = ImageDraw.Draw(img)
    z = np.clip((p[:, 2] - p[:, 2].min()) / max(np.ptp(p[:, 2]), 1e-6), 0, 1)
    r = max(1, int(0.2 * s / 2))
    for i in np.argsort(-depth):                     # far to near
        c = (int(60 + 195 * z[i]), int(200 - 120 * abs(z[i] - 0.5) * 2), int(255 - 200 * z[i]))
        d.rectangle([U[i] - r, V[i] - r, U[i] + r, V[i] + r], fill=c)
    img.save(path)
    return len(p)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else '.'
    os.makedirs(out, exist_ok=True)
    rclpy.init()
    n = Node('save_maps')
    got = {}
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    n.create_subscription(OccupancyGrid, '/map', lambda m: got.__setitem__('map', m), latched)
    n.create_subscription(OccupancyGrid, '/projected_map', lambda m: got.__setitem__('proj', m), 1)  # volatile
    n.create_subscription(MarkerArray, '/occupied_cells_vis_array', lambda m: got.__setitem__('octo', m), 5)
    end = time.time() + 15
    while time.time() < end and len(got) < 3:
        rclpy.spin_once(n, timeout_sec=0.2)
    if 'map' in got:
        occ, free = grid_png(got['map'], os.path.join(out, 'slam_map.png'))
        print(f'slam_map.png       {occ} obstacle cells, {free:.0f} m^2 free')
    if 'proj' in got:
        occ, free = grid_png(got['proj'], os.path.join(out, 'projected_map.png'))
        print(f'projected_map.png  {occ} obstacle cells')
    if 'octo' in got:
        print(f'octomap_3d.png     {octomap_png(got["octo"], os.path.join(out, "octomap_3d.png"))} occupied voxels')
    missing = {'map', 'proj', 'octo'} - set(got)
    if missing:
        print(f'not received: {sorted(missing)}')
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
