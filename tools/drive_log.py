#!/usr/bin/env python3
"""Log the car every N seconds: position, planner decision, shield, traffic rule, route state.
Saves a front-camera frame whenever the car is inside a given box (e.g. a tunnel).
  python3 tools/drive_log.py SECONDS [x0 x1 y0 y1 out_dir]"""
import json
import os
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

secs = float(sys.argv[1])
box = [float(v) for v in sys.argv[2:6]] if len(sys.argv) >= 6 else None
out = sys.argv[6] if len(sys.argv) >= 7 else None
rclpy.init()
n = Node('drive_log')
st = {}
n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('g', m.point), 5)
for t in ('/planner/status', '/safety/status', '/behaviour/state', '/route/status'):
    n.create_subscription(String, t, lambda m, t=t: st.__setitem__(t, json.loads(m.data)), 5)
n.create_subscription(Image, '/camera/image_raw', lambda m: st.__setitem__('img', m), 1)
t0, last, saved = time.time(), 0.0, 0
while time.time() - t0 < secs:
    rclpy.spin_once(n, timeout_sec=0.05)
    if 'g' not in st or time.time() - last < 3:
        continue
    last = time.time()
    g, p, s, b = st['g'], st.get('/planner/status', {}), st.get('/safety/status', {}), st.get('/behaviour/state', {})
    inside = box and box[0] < g.x < box[1] and box[2] < g.y < box[3]
    print(f"{time.time() - t0:5.0f}s ({g.x:6.1f},{g.y:7.1f}){' [IN BOX]' if inside else ''} plan v={p.get('speed')} "
          f"steer={p.get('steer')} free={p.get('free')} {p.get('reason')} w={p.get('road_width')} | shield "
          f"{s.get('reason')} {s.get('static_distance')} | rule {b.get('reason')} | route {st.get('/route/status', {}).get('state')}",
          flush=True)
    if inside and out and 'img' in st and saved < 6:
        from PIL import Image as PI
        m = st['img']
        a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 4)[:, :, [2, 1, 0]]
        os.makedirs(out, exist_ok=True)
        PI.fromarray(a).save(os.path.join(out, f'box_{saved}.png'))
        saved += 1
