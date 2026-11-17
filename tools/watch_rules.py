#!/usr/bin/env python3
"""Print traffic-rule decisions (light colour, stop/go, distance to the stop line) and the car's
position/speed whenever the decision changes.   python3 tools/watch_rules.py SECONDS"""
import json
import sys
import time

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

rclpy.init()
n = Node('watch_rules')
st = {}
n.create_subscription(String, '/behaviour/state', lambda m: st.__setitem__('b', json.loads(m.data)), 5)
n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('g', m.point), 5)
n.create_subscription(Odometry, '/wheel/odom', lambda m: st.__setitem__('v', m.twist.twist.linear.x), 5)
n.create_subscription(String, '/route/status', lambda m: st.__setitem__('r', json.loads(m.data)), 5)
t0, last, lastr = time.time(), None, None
while time.time() - t0 < float(sys.argv[1] if len(sys.argv) > 1 else 120):
    rclpy.spin_once(n, timeout_sec=0.05)
    b = st.get('b')
    if not b or 'g' not in st:
        continue
    key = (b['light'], b['reason'], b['stop'])
    if key != last:
        last = key
        print(f"t={time.time() - t0:5.1f}s ({st['g'].x:6.1f},{st['g'].y:6.1f}) v={st.get('v', 0):4.1f} m/s  "
              f"light={b['light']} {'STOP' if b['stop'] else 'go  '} reason={b['reason']} "
              f"line={b['stop_line_distance']} m", flush=True)
    r = st.get('r', {}).get('state')
    if r != lastr:
        lastr = r
        print(f"t={time.time() - t0:5.1f}s route: {r}", flush=True)
        if r == 'goal_reached':
            break
