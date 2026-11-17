#!/usr/bin/env python3
"""Driving-stability measurement for one run (used by tools/speed_sweep.sh).

  python3 tools/stability_probe.py SECONDS LABEL

Records while the car moves (> 1 m/s):
  * true lane offset: GPS + IMU against the lane map (independent of the planner): mean |offset|,
    95th percentile, % of time the body edge is over the centre line / right edge
  * zig-zag: steering command reversals per minute (swings > 0.02 rad) and steering std
  * speed: mean / std while moving, number of stops (speed < 0.3 m/s for > 2 s)
Prints one summary line.
"""
import math
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu

sys.path.insert(0, '.')
from auto_car_core.lanemap import LaneMap  # noqa: E402

lm = LaneMap('worlds/village_net/sumo.net.xml')
rclpy.init()
n = rclpy.create_node('stability_probe')
st = {}
n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('g', m.point), 10)
n.create_subscription(Imu, '/imu/data', lambda m: st.__setitem__('q', m.orientation), 10)
n.create_subscription(Odometry, '/wheel/odom', lambda m: st.__setitem__('v', m.twist.twist.linear.x), 10)
n.create_subscription(AckermannDriveStamped, '/cmd_drive', lambda m: st.__setitem__('s', m.drive.steering_angle), 10)
offs, steers, speeds, stops = [], [], [], 0
t0, last, still_since = time.time(), 0.0, None
while time.time() - t0 < float(sys.argv[1]):
    rclpy.spin_once(n, timeout_sec=0.02)
    if not all(k in st for k in ('g', 'q', 'v', 's')) or time.time() - last < 0.1:
        continue
    last = time.time()
    v = st['v']
    if v < 0.3:
        still_since = still_since or last
        if last - still_since > 2.0 and still_since > 0:
            stops += 1
            still_since = -1e9                                 # count each stop once
        continue
    if still_since is not None and still_since > 0:
        still_since = None
    elif still_since is not None and still_since < 0:
        still_since = None
    if v < 1.0:
        continue
    q = st['q']
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    c = np.array([st['g'].x - math.cos(yaw) + 1.45 * math.cos(yaw), st['g'].y - math.sin(yaw) + 1.45 * math.sin(yaw)])
    hit = lm.nearest_lane(tuple(c), yaw, max_angle=math.radians(40), include_internal=False)
    if hit is not None and hit[2] < 3.0:
        lane, s_, _ = hit
        q0, q1 = lane.at(s_), lane.at(s_ + 1.0)
        t = (q1 - q0) / max(np.hypot(*(q1 - q0)), 1e-9)
        offs.append(float((c - q0) @ np.array([-t[1], t[0]])))
    steers.append(st['s'])
    speeds.append(v)
o, s, sp = np.array(offs), np.array(steers), np.array(speeds)
if len(s) < 20:
    print(f'{sys.argv[2]}: not enough driving data (car did not move)')
    raise SystemExit
ds = np.diff(s)
ds = ds[np.abs(ds) > 1e-3]
rev = int(np.sum(np.diff(np.sign(ds)) != 0)) if len(ds) > 1 else 0
# count only reversals between swings larger than 0.02 rad (real zig-zag, not noise)
peaks, last_ext, direction = 0, s[0], 0
for x in s[1:]:
    if direction >= 0 and x < last_ext - 0.02:
        peaks += direction > 0
        direction, last_ext = -1, x
    elif direction <= 0 and x > last_ext + 0.02:
        peaks += direction < 0
        direction, last_ext = 1, x
    elif (direction > 0 and x > last_ext) or (direction < 0 and x < last_ext):
        last_ext = x
minutes = len(s) * 0.1 / 60.0
print(f'{sys.argv[2]}: speed {sp.mean():.1f}+-{sp.std():.1f} m/s (max {sp.max():.1f}), stops {stops} | '
      f'lane |offset| mean {np.abs(o).mean():.2f} m p95 {np.percentile(np.abs(o), 95):.2f} m, '
      f'over centre line {100 * (o > 0.75).mean():.0f}% over edge {100 * (o < -0.75).mean():.0f}% | '
      f'zig-zag {peaks / max(minutes, 1e-3):.1f} swings/min, steer std {s.std():.3f} rad '
      f'({len(o)} lane samples)' if len(o) else f'{sys.argv[2]}: no lane samples')
