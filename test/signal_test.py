#!/usr/bin/env python3
"""Traffic-light behaviour at the city junction (45, -45): stop on red/orange, go on green.

Drives straight through the junction via the lane map (launch with stack:=false; the test sends
the goal, then starts the planner). Records the light colour the car perceives, its speed and
the distance to the stop line, and saves the light crop at every stop/go decision for checking.

  python3 test/signal_test.py --goal 37.2 -75 --seconds 90 --save DIR
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import String


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def send_goal_until_routed(node, pub, goal, timeout=15.0):
    """Publish the goal until the route planner confirms a route (discovery can drop early messages)."""
    import json as _json
    from std_msgs.msg import String as _S
    got = {}
    sub = node.create_subscription(_S, '/route/status', lambda m: got.__setitem__('s', _json.loads(m.data)), 10)
    t0, last = time.time(), 0.0
    while time.time() - t0 < timeout and got.get('s', {}).get('state') != 'route':
        if time.time() - last > 0.5:
            pub.publish(goal)
            last = time.time()
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(sub)
    ok = got.get('s', {}).get('state') == 'route'
    print(f"route planner: {'route ready' if ok else 'NO ROUTE (goal not confirmed)'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal', type=float, nargs=2, default=(37.2, -75.0))
    ap.add_argument('--seconds', type=float, default=90)
    ap.add_argument('--save', default='/tmp')
    ap.add_argument('--delay', type=float, default=0.0, help='wait before driving (meet a different light phase)')
    ap.add_argument('--stop-sign', action='store_true',
                    help='unsignalised junction with a stop sign: expect a full stop at the line, a wait, then go '
                         '(no traffic light is required to be seen)')
    a = ap.parse_args()
    os.makedirs(a.save, exist_ok=True)
    rclpy.init()
    n = Node('signal_test')
    st = {'gps': None, 'yaw': None, 'v': 0.0, 'b': {}, 'crop': None}
    n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m.point), 10)
    n.create_subscription(Imu, '/imu/data', lambda m: st.__setitem__('yaw', yaw_of(m.orientation)), 10)
    n.create_subscription(Odometry, '/wheel/odom', lambda m: st.__setitem__('v', m.twist.twist.linear.x), 10)
    n.create_subscription(String, '/behaviour/state', lambda m: st.__setitem__('b', json.loads(m.data)), 10)
    n.create_subscription(Image, '/behaviour/light_crop', lambda m: st.__setitem__('crop', m), 2)
    st['pl'], st['sh'] = {}, {}
    n.create_subscription(String, '/planner/status', lambda m: st.__setitem__('pl', json.loads(m.data)), 10)
    n.create_subscription(String, '/safety/status', lambda m: st.__setitem__('sh', json.loads(m.data)), 10)
    from geometry_msgs.msg import PoseArray
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    n.create_subscription(PoseArray, '/route/stop_lines', lambda m: st.__setitem__('lines', [
        (p.position.x, p.position.y, yaw_of(p.orientation)) for p in m.poses]), latched)
    n.create_subscription(Odometry, '/odometry/filtered', lambda m: st.__setitem__('pose', (
        m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))), 10)
    pub = n.create_publisher(PoseStamped, '/goal_pose', 10)
    while st['gps'] is None or st['yaw'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    ox, oy = st['gps'].x - math.cos(st['yaw']), st['gps'].y - math.sin(st['yaw'])
    g = PoseStamped()
    g.header.frame_id = 'odom'
    g.pose.position.x, g.pose.position.y = a.goal[0] - ox, a.goal[1] - oy
    g.pose.orientation.w = 1.0
    send_goal_until_routed(n, pub, g)
    time.sleep(a.delay)
    planner = subprocess.Popen(['ros2', 'run', 'auto_car', 'local_planner.py', '--ros-args', '-p', 'use_sim_time:=true'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    events, last_stop, lights_seen, violations, min_line = [], None, {}, 0, math.inf
    stopped_at_line = False
    stop_heading_err = 0.0
    reasons, reached = set(), False
    last_dbg = 0.0
    held_on_green = 0
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.05)
        b = st['b']
        if not b:
            continue
        lights_seen[b.get('light')] = lights_seen.get(b.get('light'), 0) + 1
        reasons.add(b.get('reason'))
        d = b.get('stop_line_distance')
        if d is not None and d < 25:
            min_line = min(min_line, d)
            if d < -0.5 and b.get('light') == 'red':
                violations += 1
            if abs(st['v']) < 0.1 and 0 <= d < 6 and b.get('stop'):
                stopped_at_line = True
                if st.get('lines') and st.get('pose'):      # heading vs the lane at the stop line
                    x, y, yaw = st['pose']
                    lx, ly, lh = min(st['lines'], key=lambda l: math.hypot(l[0] - x, l[1] - y))
                    err = abs(math.degrees(math.atan2(math.sin(yaw - lh), math.cos(yaw - lh))))
                    stop_heading_err = max(stop_heading_err, err)
        if not b.get('stop') and abs(st['v']) < 0.1 and d is not None and 0 <= d < 6:
            held_on_green += 1
            if time.time() - last_dbg > 3:
                last_dbg = time.time()
                print(f"    held while allowed to go: light={b.get('light')} turn={b.get('turn')} lamps={b.get('lamps')} "
                      f"planner={st['pl'].get('reason')} v_plan={st['pl'].get('speed')} shield={st['sh'].get('reason')} "
                      f"obst={st['sh'].get('static_distance')}")
        if b.get('stop') != last_stop:
            last_stop = b.get('stop')
            ev = f"t={time.time() - t0:5.1f}s {'STOP' if last_stop else 'GO  '} light={b.get('light')} reason={b.get('reason')} " \
                 f"line={d} m speed={st['v']:.1f}"
            events.append(ev)
            print('  ' + ev)
            if st['crop'] is not None:
                c = st['crop']
                from PIL import Image as PI
                img = np.frombuffer(bytes(c.data), np.uint8).reshape(c.height, c.width, 3)[:, :, ::-1]
                PI.fromarray(img).resize((c.width * 8, c.height * 8), PI.NEAREST).save(
                    os.path.join(a.save, f"light_{len(events):02d}_{'stop' if last_stop else 'go'}.png"))
        if math.hypot(st['gps'].x - a.goal[0], st['gps'].y - a.goal[1]) < 8:
            print(f'  goal reached at t={time.time() - t0:.0f}s')
            reached = True
            break
    planner.terminate()
    print(f'lights perceived (frames): {lights_seen}')
    print(f'stopped at the stop line: {stopped_at_line}, closest to line {min_line:.1f} m, red-light violations: {violations}, '
          f'frames held at the line while allowed to go: {held_on_green}')
    print(f'heading error while stopped at the line: {stop_heading_err:.1f} deg (car straight in its lane)')
    if a.stop_sign:
        checks = [('no crossing on red', violations == 0), ('full stop at the stop line', stopped_at_line),
                  ('stopped straight (heading error <= 5 deg)', stop_heading_err <= 5.0),
                  ('waited at the stop sign', 'stop_sign_wait' in reasons), ('goal reached after the stop', reached)]
    else:
        checks = [('no crossing on red', violations == 0),
                  ('stopped straight (heading error <= 5 deg)', not stopped_at_line or stop_heading_err <= 5.0),
                  ('traffic light colour perceived', any(k in lights_seen for k in ('red', 'orange', 'green')))]
    for name, c in checks:
        print(f"  [{'PASS' if c else 'FAIL'}] {name}")
    ok = all(c for _, c in checks)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
