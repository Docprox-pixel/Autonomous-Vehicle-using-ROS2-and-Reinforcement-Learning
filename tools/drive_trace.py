#!/usr/bin/env python3
"""Trace what the planner does while driving, at 5 Hz, and summarise the zig-zag sources.

  python3 tools/drive_trace.py SECONDS [OUT.txt]

Per sample: speed, lateral error to the route (body centre, + = left) and heading error, the
planner's steering and the steering actually sent (/cmd_drive), the planner reason, the in-lane
nudge and the active speed limit. Summary: steering swings/min, how often the nudge moves, the
correlation between nudge changes and steering changes, and when the car was > 0.5 m off.
"""
import json
import math
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

dur = float(sys.argv[1]) if len(sys.argv) > 1 else 60.0
out = open(sys.argv[2], 'w') if len(sys.argv) > 2 else None
rclpy.init()
n = rclpy.create_node('drive_trace')
st = {}
lat_q = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


n.create_subscription(Odometry, '/odometry/filtered', lambda m: st.__setitem__(
    'o', (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation), m.twist.twist.linear.x)), 20)
n.create_subscription(Path, '/route/path', lambda m: st.__setitem__(
    'p', np.array([[q.pose.position.x, q.pose.position.y] for q in m.poses])), lat_q)
n.create_subscription(String, '/planner/status', lambda m: st.__setitem__('s', json.loads(m.data)), 10)
n.create_subscription(AckermannDriveStamped, '/cmd_drive', lambda m: st.__setitem__('c', m.drive.steering_angle), 10)
n.create_subscription(String, '/behaviour/state', lambda m: st.__setitem__('b', json.loads(m.data)), 10)
n.create_subscription(String, '/safety/status', lambda m: st.__setitem__('sf', json.loads(m.data)), 10)
rows, t0, last = [], time.time(), 0.0
while time.time() - t0 < dur:
    rclpy.spin_once(n, timeout_sec=0.01)
    if time.time() - last < 0.2 or not all(k in st for k in ('o', 'p', 's', 'c')) or len(st['p']) < 3:
        continue
    last = time.time()
    x, y, yaw, v = st['o']
    P = st['p']
    oc = np.array([x + 1.45 * math.cos(yaw), y + 1.45 * math.sin(yaw)])
    i = int(np.argmin(np.hypot(*(P - oc).T)))
    j, k = min(i + 2, len(P) - 1), max(i - 2, 0)
    t = P[j] - P[k]
    t /= max(np.hypot(*t), 1e-9)
    e = float((oc - P[i]) @ np.array([-t[1], t[0]]))
    he = math.degrees(math.atan2(math.sin(yaw - math.atan2(t[1], t[0])), math.cos(yaw - math.atan2(t[1], t[0]))))
    s = st['s']
    row = (time.time() - t0, v, e, he, float(s.get('steer', 0.0)), st['c'], float(s.get('nudge') or 0.0),
           s.get('reason'), s.get('limit'), s.get('nudge_why', ''), x, y)
    rows.append(row)
    line = (f'{row[0]:5.1f} v={v:4.1f} lat={e:+.2f} head={he:+5.1f} plan={row[4]:+.2f} cmd={row[5]:+.2f} '
            f'nudge={row[6]:+.2f}{row[9] or ""} reason={row[7]} limit={row[8]} odom=({x:.1f},{y:.1f})'
            f' rules={st.get("b", {}).get("reason")}/{st.get("b", {}).get("stop_line_distance")}'
            f' shield={st.get("sf", {}).get("reason")}/{st.get("sf", {}).get("static_distance")}')
    if out:
        out.write(line + '\n')
R = np.array([r[:7] for r in rows]) if rows else np.zeros((0, 7))
if len(R) < 10:
    raise SystemExit('not enough samples')
t, v, lat, head, ps, cs, nud = R.T
mv = v > 1.0
sw, ext, d = 0, ps[0], 0
for s_ in ps:
    if d >= 0 and s_ < ext - 0.02:
        sw += d > 0
        d, ext = -1, s_
    elif d <= 0 and s_ > ext + 0.02:
        sw += d < 0
        d, ext = 1, s_
    elif (d > 0 and s_ > ext) or (d < 0 and s_ < ext):
        ext = s_
dn = np.abs(np.diff(nud))
print(f'{len(t)} samples, moving {mv.mean() * 100:.0f}%: steering swings {sw / (len(t) * 0.2 / 60):.1f}/min | '
      f'|lat| mean {np.abs(lat[mv]).mean():.2f} m max {np.abs(lat[mv]).max():.2f} m, heading std {head[mv].std():.1f} deg')
print(f'nudge active {100 * (np.abs(nud[mv]) > 0.02).mean():.0f}% of moving time, changes per sample mean {dn.mean():.3f} m; '
      f'corr(|d nudge|, |d steer|) = {np.corrcoef(dn, np.abs(np.diff(ps)))[0, 1]:.2f}')
big = t[mv & (np.abs(lat) > 0.5)]
print('off > 0.5 m at t =', np.round(big[::3], 1)[:25])
