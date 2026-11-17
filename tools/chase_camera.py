#!/usr/bin/env python3
"""Make the Webots 3D view follow the ego car (chase camera), e.g. for screen recording.

  python3 tools/chase_camera.py worlds/village.wbt [--back 14] [--up 6] [--pitch 0.3] [--type "Mounted Shot"]

Sets the world's Viewpoint to `follow "vehicle"` (the ego car's name) and places it behind and
above the car's start pose, looking at it. followType:
  "Mounted Shot"  - rigidly attached: always behind the car, turns with it (best for recording)
  "Tracking Shot" - follows the position only, keeps its own direction
  "Pan and Tilt Shot" - stays put, turns to keep the car in view
In Webots: File > Make Movie... records the 3D view.
"""
import argparse
import math
import re


def axis_angle(yaw, pitch):
    """Rotation Rz(yaw) * Ry(pitch) as Webots axis-angle (x y z angle)."""
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    R = [[cy * cp, -sy, cy * sp], [sy * cp, cy, sy * sp], [-sp, 0.0, cp]]
    ang = math.acos(max(-1.0, min(1.0, (R[0][0] + R[1][1] + R[2][2] - 1) / 2)))
    if ang < 1e-6:
        return 0.0, 0.0, 1.0, 0.0
    x, y, z = R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]
    n = math.sqrt(x * x + y * y + z * z)
    return x / n, y / n, z / n, ang


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('world')
    ap.add_argument('--back', type=float, default=14.0)
    ap.add_argument('--up', type=float, default=6.0)
    ap.add_argument('--pitch', type=float, default=0.32, help='rad, looking down')
    ap.add_argument('--type', default='Mounted Shot')
    a = ap.parse_args()
    text = open(a.world).read()
    car = re.search(r'BmwX5 \{\n  translation ([-\d.e]+) ([-\d.e]+) ([-\d.e]+)\n  rotation [-\d.e]+ [-\d.e]+ (-?[\d.e]+) ([-\d.e]+)', text)
    if not car:
        raise SystemExit('ego car (BmwX5) not found')
    x, y = float(car.group(1)), float(car.group(2))
    yaw = float(car.group(5)) * (1 if float(car.group(4)) >= 0 else -1)
    px, py, pz = x - a.back * math.cos(yaw), y - a.back * math.sin(yaw), a.up
    ax, ay, az, ang = axis_angle(yaw, a.pitch)
    vp = (f'Viewpoint {{\n  orientation {ax:.5f} {ay:.5f} {az:.5f} {ang:.5f}\n  position {px:.3f} {py:.3f} {pz:.3f}\n'
          f'  near 1\n  follow "vehicle"\n  followType "{a.type}"\n}}\n')
    text, n = re.subn(r'(?ms)^Viewpoint \{\n.*?^\}\n', vp, text, count=1)
    open(a.world, 'w').write(text)
    print(f'{a.world}: chase camera ({a.type}) {a.back} m behind, {a.up} m up' if n else 'no Viewpoint found')


if __name__ == '__main__':
    main()
