#!/usr/bin/env python3
"""Log every moment the car's body centre is > LIMIT m LEFT of its lane centre (toward the other
lane), with position, speed, planner reason and steering.  python3 tools/lane_excursions.py SECONDS [LIMIT]"""
import json, math, sys, time
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
rclpy.init(); n = Node('lane_excursions'); st = {}
q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
n.create_subscription(Path, '/route/path', lambda m: st.__setitem__('path', np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])), q)
n.create_subscription(Odometry, '/odometry/filtered', lambda m: st.__setitem__('o', m), 20)
n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('g', m.point), 5)
n.create_subscription(String, '/planner/status', lambda m: st.__setitem__('pl', json.loads(m.data)), 5)
limit = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
t0, last, vals, events = time.time(), 0, [], []
while time.time() - t0 < float(sys.argv[1]):
    rclpy.spin_once(n, timeout_sec=0.05)
    if 'o' not in st or st.get('path') is None or len(st['path']) < 3 or time.time() - last < 0.2:
        continue
    last = time.time()
    o = st['o'].pose.pose; v = st['o'].twist.twist.linear.x
    if v < 0.5:
        continue
    yaw = math.atan2(2 * o.orientation.w * o.orientation.z, 1 - 2 * o.orientation.z ** 2)
    c = np.array([o.position.x + 1.45 * math.cos(yaw), o.position.y + 1.45 * math.sin(yaw)])
    P = st['path']; i = int(np.argmin(np.hypot(*(P - c).T)))
    t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]; t /= max(np.hypot(*t), 1e-9)
    off = float((c - P[i]) @ np.array([-t[1], t[0]]))
    # route curvature here (deg over +-6 m)
    a, b = P[max(i - 6, 0)], P[min(i + 6, len(P) - 1)]
    h1 = math.atan2(*(P[i] - a)[::-1]) if np.hypot(*(P[i] - a)) > 0.5 else 0.0
    h2 = math.atan2(*(b - P[i])[::-1]) if np.hypot(*(b - P[i])) > 0.5 else 0.0
    turn = math.degrees(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1)))
    vals.append(off)
    if off > limit:
        g = st.get('g'); pl = st.get('pl', {})
        events.append((round(time.time() - t0), round(g.x, 1) if g else None, round(g.y, 1) if g else None, round(off, 2),
                       round(v, 1), round(turn), pl.get('reason'), pl.get('steer')))
v = np.array(vals)
print(f'{len(v)} samples: mean {v.mean():+.2f} m, >{limit} m left: {100 * (v > limit).mean():.1f}% of the time, max left {v.max():+.2f} m')
for e in events[::max(1, len(events) // 25)]:
    print('  t=%ss world (%s,%s) left %.2f m  v=%.1f  route turn %+d deg  planner %s steer %s' % e)
