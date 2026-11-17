#!/usr/bin/env python3
"""TRUE lane position from GPS + map (independent of odometry): signed offset of the car body centre
from the centre of the nearest same-direction lane (+ = left, toward the centre line), and the
offset the stack believes (odometry vs route).  python3 tools/true_lane_offset.py WORLD_NET SECONDS"""
import math, sys, time
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
sys.path.insert(0, '.')
from auto_car_core.lanemap import LaneMap
m = LaneMap(sys.argv[1])
rclpy.init(); n = Node('true_lane'); st = {}
n.create_subscription(PointStamped, '/gps/position', lambda mm: st.__setitem__('g', mm.point), 10)
n.create_subscription(Imu, '/imu/data', lambda mm: st.__setitem__('q', mm.orientation), 10)
n.create_subscription(Odometry, '/wheel/odom', lambda mm: st.__setitem__('v', mm.twist.twist.linear.x), 10)
true_off, t0, last = [], time.time(), 0
while time.time() - t0 < float(sys.argv[2]):
    rclpy.spin_once(n, timeout_sec=0.05)
    if not all(k in st for k in ('g', 'q', 'v')) or time.time() - last < 0.25 or st['v'] < 1.0:
        continue
    last = time.time()
    q = st['q']; yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    bx, by = st['g'].x - math.cos(yaw), st['g'].y - math.sin(yaw)          # base_link (GPS on roof mount 1 m ahead)
    c = np.array([bx + 1.45 * math.cos(yaw), by + 1.45 * math.sin(yaw)])
    hit = m.nearest_lane(tuple(c), yaw, max_angle=math.radians(40), include_internal=False)
    if hit is None or hit[2] > 3.0:
        continue
    lane, s_, d_ = hit
    q0 = lane.at(s_); q1 = lane.at(s_ + 1.0); t = (q1 - q0) / max(np.hypot(*(q1 - q0)), 1e-9)
    true_off.append(float((c - q0) @ np.array([-t[1], t[0]])))
v = np.array(true_off)
if len(v):
    print(f'{len(v)} samples on lanes: TRUE offset mean {v.mean():+.2f} m (+ = toward centre line), '
          f'|mean| {np.abs(v).mean():.2f}; body edge over the centre line (offset > 0.75 m): {100 * (v > 0.75).mean():.0f}% ; '
          f'over the right edge (< -0.75 m): {100 * (v < -0.75).mean():.0f}%')
