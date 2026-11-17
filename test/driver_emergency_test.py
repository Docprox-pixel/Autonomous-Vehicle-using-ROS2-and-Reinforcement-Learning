#!/usr/bin/env python3
"""Driver emergency: the driver stops responding -> warnings -> the car drives to the nearest hospital.

World: worlds/village.wbt (has 'Village Hospital'). Launch with stack:=false (the test starts the
planner) and driver_monitor:=true (default).
  1. the car drives toward --goal while the 'driver' responds every 3 s for --attentive seconds
  2. the driver goes silent
  3. expect WARNING ~10 s, ALARM ~20 s, EMERGENCY ~30 s after the last response (sim time),
     hazard lights, an emergency call, destination = hospital, then ARRIVED, stopped near it.
"""
import argparse
import json
import math
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import Bool, String


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--goal', default='Museum', help='first destination (place name or kind)')
    ap.add_argument('--attentive', type=float, default=20.0, help='s of driver responses before going silent')
    ap.add_argument('--seconds', type=float, default=900.0, help='wall-clock budget')
    a = ap.parse_args()
    rclpy.init()
    n = Node('driver_emergency_test')
    st = {'driver': {}, 'hazard': None, 'call': None, 'dest': None, 'v': 0.0, 'gps': None}
    events = []

    def sim_now():
        return n.get_clock().now().nanoseconds * 1e-9

    def on_driver(m):
        d = json.loads(m.data)
        if d.get('state') != st['driver'].get('state'):
            events.append((d.get('state'), d.get('seconds_without_response'), time.time()))
            print(f"  driver state -> {d.get('state')} after {d.get('seconds_without_response')} s without response"
                  f"{'  destination ' + str(d.get('destination')) if d.get('destination') else ''}", flush=True)
        st['driver'] = d
    n.create_subscription(String, '/driver/status', on_driver, 10)
    n.create_subscription(Bool, '/vehicle/hazard', lambda m: st.__setitem__('hazard', m.data), 5)
    n.create_subscription(String, '/driver/emergency_call', lambda m: st.__setitem__('call', json.loads(m.data)), 5)
    n.create_subscription(String, '/route/destination', lambda m: st.__setitem__('dest', json.loads(m.data)), 5)
    n.create_subscription(Odometry, '/wheel/odom', lambda m: st.__setitem__('v', m.twist.twist.linear.x), 10)
    n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m.point), 10)
    pub_in = n.create_publisher(String, '/driver/input', 10)
    pub_go = n.create_publisher(String, '/route/go_to', 10)
    while st['gps'] is None or not st['driver']:
        rclpy.spin_once(n, timeout_sec=0.1)
    t0 = time.time()
    while time.time() - t0 < 10 and (st['dest'] or {}).get('query') != a.goal.lower():
        pub_go.publish(String(data=a.goal))
        pub_in.publish(String(data='steering_touch'))
        rclpy.spin_once(n, timeout_sec=0.5)
    print(f"first destination: {st['dest']}")
    planner = subprocess.Popen(['ros2', 'run', 'auto_car', 'local_planner.py', '--ros-args', '-p', 'use_sim_time:=true'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    s0, last_touch, silent_from = sim_now(), -1e9, None
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.05)
        ts = sim_now() - s0
        if ts < a.attentive:
            if ts - last_touch > 3.0:
                pub_in.publish(String(data='steering_touch'))
                last_touch = ts
        elif silent_from is None:
            silent_from = ts
            print(f'  t={ts:.0f}s (sim): driver goes silent at ({st["gps"].x:.0f}, {st["gps"].y:.0f})', flush=True)
        if st['driver'].get('state') == 'ARRIVED' and abs(st['v']) < 0.1:
            break
    planner.terminate()
    states = [e[0] for e in events]
    hosp = st['dest'] if st['dest'] and st['dest'].get('query') == 'hospital' else None
    d_h = math.hypot(st['gps'].x - hosp['x'], st['gps'].y - hosp['y']) if hosp and hosp.get('found') else math.inf
    print(f"\nstates: {' -> '.join(s for s in states if s)}")
    print(f"emergency call: {st['call']}")
    print(f"hazard lights: {st['hazard']}, destination: {hosp and hosp.get('name')}, "
          f"final position ({st['gps'].x:.0f}, {st['gps'].y:.0f}), {d_h:.0f} m from the hospital, speed {st['v']:.1f} m/s")
    checks = [('warning then alarm then emergency', all(s in states for s in ('WARNING', 'ALARM', 'EMERGENCY'))
               and states.index('WARNING') < states.index('ALARM') < states.index('EMERGENCY')),
              ('hazard lights on', st['hazard'] is True),
              ('emergency call made', st['call'] is not None),
              ('destination switched to a hospital', hosp is not None and hosp.get('found')),
              ('arrived and stopped at the hospital (< 30 m)', 'ARRIVED' in states and abs(st['v']) < 0.1 and d_h < 30)]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(c for _, c in checks)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
