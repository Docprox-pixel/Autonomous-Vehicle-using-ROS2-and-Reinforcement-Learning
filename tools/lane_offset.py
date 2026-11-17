#!/usr/bin/env python3
"""Measure the car's signed offset from its lane centre (the route) while driving.
+ = left of the lane centre (toward the oncoming lane).  python3 tools/lane_offset.py SECONDS"""
import math, sys, time
import numpy as np
import rclpy
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
rclpy.init(); n = Node('lane_offset'); st = {}
q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_subscription(Path, '/route/path', lambda m: st.__setitem__('path', np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])), q)
n.create_subscription(Odometry, '/odometry/filtered', lambda m: st.__setitem__('o', m), 20)
vals, t0, last = [], time.time(), 0
while time.time() - t0 < float(sys.argv[1]):
    rclpy.spin_once(n, timeout_sec=0.05)
    if 'o' not in st or st.get('path') is None or len(st['path']) < 3 or time.time() - last < 0.2:
        continue
    last = time.time()
    o = st['o'].pose.pose; v = st['o'].twist.twist.linear.x
    if v < 1.0:
        continue
    yaw = math.atan2(2 * o.orientation.w * o.orientation.z, 1 - 2 * o.orientation.z ** 2)
    c = np.array([o.position.x + 1.45 * math.cos(yaw), o.position.y + 1.45 * math.sin(yaw)])   # body centre
    P = st['path']; i = int(np.argmin(np.hypot(*(P - c).T)))
    t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]; t /= max(np.hypot(*t), 1e-9)
    vals.append(float((c - P[i]) @ np.array([-t[1], t[0]])))
v = np.array(vals)
if len(v):
    print(f'{len(v)} samples while moving: mean offset {v.mean():+.2f} m (+ = left), |offset| mean {np.abs(v).mean():.2f}, '
          f'left of centre {100 * (v > 0.3).mean():.0f}% / right {100 * (v < -0.3).mean():.0f}% / centred {100 * (np.abs(v) <= 0.3).mean():.0f}%, '
          f'max left {v.max():+.2f} m')
else:
    print('car did not move')
