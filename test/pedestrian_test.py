#!/usr/bin/env python3
"""Pedestrian on the crosswalk ahead: detected as a person, car slows, keeps its distance.

World: worlds/city_pedestrian.wbt (a pedestrian walks back and forth across our lanes at y = -26).
Launch with stack:=false; the test sends a goal just before the junction, then starts the planner.
"""
import json
import math
import subprocess
import sys
import time

import numpy as np
import rclpy
from auto_car.msg import TrackArray
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String

GOAL = (36.8, -31.0)    # just past the crosswalk (y = -26), before the junction's stop line
GOAL_TOL = 3.0          # m: 'reached' only after crossing the person's path


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def body_gap(x, y):
    gx = max(-1.15 - x, 0.0, x - 4.05)
    gy = max(abs(y) - 1.0, 0.0)
    return math.hypot(gx, gy)


def main():
    rclpy.init()
    n = Node('pedestrian_test')
    st = {'gps': None, 'yaw': None, 'v': 0.0, 'b': {}, 'tracks': None, 'route': None}
    n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m.point), 10)
    n.create_subscription(Imu, '/imu/data', lambda m: st.__setitem__('yaw', yaw_of(m.orientation)), 10)
    n.create_subscription(Odometry, '/wheel/odom', lambda m: st.__setitem__('v', m.twist.twist.linear.x), 10)
    n.create_subscription(String, '/behaviour/state', lambda m: st.__setitem__('b', json.loads(m.data)), 10)
    n.create_subscription(TrackArray, '/perception/tracks_labeled', lambda m: st.__setitem__('tracks', m), 10)
    n.create_subscription(String, '/route/status', lambda m: st.__setitem__('route', json.loads(m.data)), 10)
    pub = n.create_publisher(PoseStamped, '/goal_pose', 10)
    while st['gps'] is None or st['yaw'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    ox, oy = st['gps'].x - math.cos(st['yaw']), st['gps'].y - math.sin(st['yaw'])
    g = PoseStamped()
    g.header.frame_id = 'odom'
    g.pose.position.x, g.pose.position.y = GOAL[0] - ox, GOAL[1] - oy
    g.pose.orientation.w = 1.0
    t0 = time.time()
    while time.time() - t0 < 15 and (st['route'] or {}).get('state') != 'route':
        pub.publish(g)
        rclpy.spin_once(n, timeout_sec=0.3)
    # Synchronise with the walking person (1 m/s, x 33 <-> 46 across our lane): start driving when
    # they are at the far (east) end walking back, so car (~9 s to the crosswalk) and person meet.
    # Without this the encounter depended on launch timing and the 'slowed' check was often vacuous.
    tw = time.time()
    while time.time() - tw < 40:
        rclpy.spin_once(n, timeout_sec=0.05)
        tr = st['tracks']
        ppl = [t for t in tr.tracks if t.label == 'person'] if tr is not None else []
        if any(t.position.y > 4.5 and t.velocity.y < -0.3 for t in ppl):
            break
    print(f'  person at the far end, walking back: start driving (waited {time.time() - tw:.0f}s)')
    planner = subprocess.Popen(['ros2', 'run', 'auto_car', 'local_planner.py', '--ros-args', '-p', 'use_sim_time:=true'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    person_frames, near_speeds, min_gap, reached, last, flagged, yielded = 0, [], math.inf, None, 0.0, 0, 0
    while time.time() - t0 < 60:
        rclpy.spin_once(n, timeout_sec=0.05)
        tr = st['tracks']
        if tr is not None:
            people = [t for t in tr.tracks if t.label == 'person']
            if people:
                person_frames += 1
            for t in people:
                gap = body_gap(t.position.x, t.position.y) - max(t.size.x, t.size.y) / 2
                min_gap = min(min_gap, gap)
                if gap < 10.0 and t.position.x > 3.0:   # approaching a person ahead (not one already passed)
                    near_speeds.append(st['v'])
        flagged = flagged + 1 if st['b'].get('pedestrian_near') else flagged
        if st['b'].get('reason') == 'yield_pedestrian':
            yielded += 1
        if time.time() - last > 3:
            last = time.time()
            print(f"  t={time.time() - t0:4.1f}s pos ({st['gps'].x:5.1f},{st['gps'].y:6.1f}) v {st['v']:.1f} "
                  f"pedestrian_near={st['b'].get('pedestrian_near')} limit={st['b'].get('speed_limit')}")
        if math.hypot(st['gps'].x - GOAL[0], st['gps'].y - GOAL[1]) < GOAL_TOL:
            reached = time.time() - t0
            break
    planner.terminate()
    vmax = max(near_speeds) if near_speeds else float('nan')
    print(f'person-labelled frames {person_frames}, closest person gap {min_gap:.2f} m, '
          f'rules flagged pedestrian_near in {flagged} frames, max speed approaching a person < 10 m ahead: {vmax:.1f} m/s, goal {"reached at %.0fs" % reached if reached else "NOT reached"}, '
          f'frames yielding {yielded}')
    checks = [('person detected', person_frames > 10), ('slowed near the person (<= 3.5 m/s)', near_speeds and vmax <= 3.5),
              ('yielded (braked for the person in the path)', yielded > 5),
              ('kept >= 2 m from the person', min_gap >= 2.0), ('reached the goal', reached is not None)]
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(ok for _, ok in checks)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
