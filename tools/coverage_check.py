#!/usr/bin/env python3
"""Lidar blind-zone analysis (pure geometry, no simulator needed).

For every 0.2 m cell around the car, an object of height --height (default 0.5 m, a cone or a
small child) standing on the road is "seen" if at least one lidar ray hits it:
  * the cell is inside the lidar's horizontal FOV and [min, max] range,
  * some layer's ray passes between ground and the object's top at that distance,
  * the object is wide enough not to fall between two neighbouring rays,
  * the ray is not blocked by the car's own body (car_box).

  python3 tools/coverage_check.py                      # layout from config/sensor_layout.yaml
  python3 tools/coverage_check.py --legacy             # the original single 11.5 deg lidar
  python3 tools/coverage_check.py --png out.png        # also save a coverage image
"""
import argparse
import math
import os

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
LEGACY = {  # worlds/city.wbt before the blind-spot fix
    'ground_z': -0.32,
    'car_box': {'x': [-1.15, 4.05], 'y': [-1.0, 1.0], 'z': [-0.32, 1.45]},
    'lidars': [{'name': 'lidar', 'xyz': [1.0, 0.0, 2.95], 'yaw': 0.0, 'hfov': 6.28318, 'vfov': 0.2,
                'layers': 16, 'resolution': 1024, 'range': [0.1, 100.0]}],
}


def blocked_by_car(p0, p1, box, samples=40):
    t = np.linspace(0.05, 0.95, samples)[:, None]
    pts = p0 + t * (p1 - p0)
    inside = ((pts[:, 0] > box['x'][0]) & (pts[:, 0] < box['x'][1]) &
              (pts[:, 1] > box['y'][0]) & (pts[:, 1] < box['y'][1]) &
              (pts[:, 2] > box['z'][0]) & (pts[:, 2] < box['z'][1]))
    return bool(inside.any())


def seen_by(lid, x, y, ground, height, width, box):
    lx, ly, lz = lid['xyz']
    dx, dy = x - lx, y - ly
    d = math.hypot(dx, dy)
    if not (max(lid['range'][0], lid.get('near_blank', 0.0)) <= d <= lid['range'][1]):
        return False
    rel = math.atan2(math.sin(math.atan2(dy, dx) - lid['yaw']), math.cos(math.atan2(dy, dx) - lid['yaw']))
    if abs(rel) > lid['hfov'] / 2:
        return False
    if width / max(d, 1e-6) < lid['hfov'] / lid['resolution']:   # falls between two rays
        return False
    n = lid['layers']
    elevs = [lid['vfov'] / 2 - i * lid['vfov'] / (n - 1) for i in range(n)] if n > 1 else [0.0]
    for e in elevs:
        z = lz + d * math.tan(e)
        if ground <= z <= ground + height:
            if not blocked_by_car(np.array([lx, ly, lz]), np.array([x, y, z]), box):
                return True
    return False


def analyse(layout, height, width, extent, res):
    box, ground = layout['car_box'], layout['ground_z']
    xs = np.arange(box['x'][0] - extent, box['x'][1] + extent + 1e-9, res)
    ys = np.arange(-extent - 1.0, extent + 1.0 + 1e-9, res)
    seen = np.zeros((len(ys), len(xs)), bool)
    gap = np.zeros(seen.shape)         # distance from the car body (m)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            gx = max(box['x'][0] - x, 0, x - box['x'][1])
            gy = max(box['y'][0] - y, 0, y - box['y'][1])
            gap[j, i] = math.hypot(gx, gy)
            if gap[j, i] < 0.3:        # touching the car: not a meaningful cell
                continue
            seen[j, i] = any(seen_by(l, x, y, ground, height, width, box) for l in layout['lidars'])
    return xs, ys, seen, gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layout', default=os.path.join(HERE, '..', 'config', 'sensor_layout.yaml'))
    ap.add_argument('--legacy', action='store_true')
    ap.add_argument('--height', type=float, default=0.5)
    ap.add_argument('--width', type=float, default=0.3)
    ap.add_argument('--extent', type=float, default=12.0)
    ap.add_argument('--res', type=float, default=0.2)
    ap.add_argument('--png')
    a = ap.parse_args()

    layout = LEGACY if a.legacy else yaml.safe_load(open(a.layout))
    xs, ys, seen, gap = analyse(layout, a.height, a.width, a.extent, a.res)
    cell = a.res ** 2
    name = 'legacy single lidar' if a.legacy else os.path.basename(a.layout)
    print(f'{name}: object {a.height} m tall x {a.width} m wide, {len(layout["lidars"])} lidar(s)')
    for r in (2, 5, 10):
        zone = (gap >= 0.3) & (gap <= r)
        blind = zone & ~seen
        print(f'  within {r:>2} m of the car body: {100 * seen[zone].mean():5.1f}% covered, '
              f'blind area {blind.sum() * cell:6.1f} m^2')
    if a.png:
        from PIL import Image
        img = np.full(seen.shape + (3,), 255, np.uint8)
        img[(gap >= 0.3) & ~seen] = (230, 60, 60)        # blind: red
        img[seen] = (120, 200, 120)                      # seen: green
        img[gap < 0.3] = (40, 40, 40)                    # car
        for r in (5, 10):                                # range rings
            ring = np.abs(gap - r) < a.res / 2
            img[ring] = (60, 60, 200)
        Image.fromarray(np.flipud(img)).resize((seen.shape[1] * 4, seen.shape[0] * 4), Image.NEAREST).save(a.png)
        print(f'  image: {a.png}  (green seen, red blind, dark car, blue rings 5/10 m)')


if __name__ == '__main__':
    main()
