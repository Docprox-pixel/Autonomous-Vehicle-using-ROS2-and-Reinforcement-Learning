#!/usr/bin/env python3
"""Measure the car's real braking: accelerate to --speed, command 0, record decel and distance.

The safety shield's a_brake must not exceed what the vehicle can do, otherwise it brakes too late.
Uses /cmd_drive directly (bypasses the shield) and sim time from message stamps.
"""
import argparse
import math
import sys
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--speed', type=float, default=8.0)
    a = ap.parse_args()
    rclpy.init()
    n = Node('braking_test')
    st = {'v': 0.0, 't': 0.0, 'gps': None}

    def on_odom(m):
        st['v'], st['t'] = m.twist.twist.linear.x, m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
    n.create_subscription(Odometry, '/wheel/odom', on_odom, 20)
    n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m.point), 10)
    pub = n.create_publisher(AckermannDriveStamped, '/cmd_drive', 10)

    def drive(speed, until):
        t0 = time.time()
        while time.time() - t0 < 30 and not until():
            c = AckermannDriveStamped()
            c.drive.speed = speed
            pub.publish(c)
            rclpy.spin_once(n, timeout_sec=0.02)

    while st['gps'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    drive(a.speed, lambda: st['v'] > a.speed - 0.1)
    v0, t0, p0 = st['v'], st['t'], (st['gps'].x, st['gps'].y)
    trace = []

    def stopped():
        trace.append((st['t'] - t0, st['v']))
        return st['v'] < 0.05
    drive(0.0, stopped)
    dt = st['t'] - t0
    dist = math.hypot(st['gps'].x - p0[0], st['gps'].y - p0[1])
    decel = v0 / dt if dt > 0 else float('nan')
    # time to first deceleration (actuation latency)
    lat = next((t for t, v in trace if v < v0 - 0.2), float('nan'))
    print(f'from {v0:.1f} m/s: stopped in {dt:.2f} s over {dist:.2f} m  ->  mean decel {decel:.2f} m/s^2, '
          f'latency to start braking {lat:.2f} s')
    print(f'shield a_brake should be <= {0.85 * decel:.1f} m/s^2 (85% of measured)')
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0)


if __name__ == '__main__':
    main()
