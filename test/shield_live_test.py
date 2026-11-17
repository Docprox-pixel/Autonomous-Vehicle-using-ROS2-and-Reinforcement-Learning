#!/usr/bin/env python3
"""Live safety-shield test: command the car straight into a box and check it stops in time.

A 0.6 x 1.8 x 1.0 m box is spawned --ahead m in front of the bumper (via Ros2Supervisor).
/cmd_drive_raw asks for --speed m/s straight ahead (a planner that would crash). The shield
must stop the car with a gap between --min-gap and --max-gap. Ground truth from GPS + IMU.
"""
import argparse
import json
import math
import sys
import time

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from webots_ros2_msgs.srv import SpawnNodeFromString

FRONT, GPS_X, BOX_LEN = 4.05, 1.0, 0.6


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ahead', type=float, default=25.0)
    ap.add_argument('--speed', type=float, default=8.0)
    ap.add_argument('--min-gap', type=float, default=0.8)
    ap.add_argument('--max-gap', type=float, default=6.0)
    a = ap.parse_args()

    rclpy.init()
    n = Node('shield_live_test')
    n.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    st = {'gps': None, 'imu': None, 'v': 0.0, 'status': {}}
    n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m), 10)
    n.create_subscription(Imu, '/imu/data', lambda m: st.__setitem__('imu', m), 10)
    n.create_subscription(Odometry, '/wheel/odom', lambda m: st.__setitem__('v', m.twist.twist.linear.x), 10)
    n.create_subscription(String, '/safety/status', lambda m: st.__setitem__('status', json.loads(m.data)), 10)
    pub = n.create_publisher(AckermannDriveStamped, '/cmd_drive_raw', 10)
    spawn = n.create_client(SpawnNodeFromString, '/Ros2Supervisor/spawn_node_from_string')
    remove = n.create_publisher(String, '/Ros2Supervisor/remove_node', 10)

    end = time.time() + 30
    while time.time() < end and (st['gps'] is None or st['imu'] is None):
        rclpy.spin_once(n, timeout_sec=0.1)
    spawn.wait_for_service(timeout_sec=10)

    def front_xy():
        g, yaw = st['gps'].point, yaw_of(st['imu'].orientation)
        return g.x + (FRONT - GPS_X) * math.cos(yaw), g.y + (FRONT - GPS_X) * math.sin(yaw), yaw

    fx, fy, yaw = front_xy()
    c, s = math.cos(yaw), math.sin(yaw)
    centre = a.ahead + BOX_LEN / 2
    bx, by = fx + c * centre, fy + s * centre
    ground = st['gps'].point.z - (1.95 + 0.32)
    node = (f'Solid {{ name "shield_box" translation {bx:.3f} {by:.3f} {ground + 0.5:.3f} rotation 0 0 1 {yaw:.4f} '
            f'children [ Shape {{ appearance PBRAppearance {{ baseColor 0.9 0.1 0.1 }} '
            f'geometry Box {{ size {BOX_LEN} 1.8 1.0 }} }} ] }}')
    fut = spawn.call_async(SpawnNodeFromString.Request(data=node))
    rclpy.spin_until_future_complete(n, fut, timeout_sec=5)
    if not (fut.result() and fut.result().success):
        print('FAIL: box could not be spawned')
        return 1
    print(f'Box {a.ahead:.0f} m ahead. Commanding {a.speed} m/s straight into it via /cmd_drive_raw ...')

    def gap():
        x, y, _ = front_xy()
        return (bx - x) * c + (by - y) * s - BOX_LEN / 2   # along the original heading

    min_gap, max_v, reasons, stopped_at = math.inf, 0.0, set(), None
    t0 = last_print = time.time()
    while time.time() - t0 < 25:
        cmd = AckermannDriveStamped()
        cmd.drive.speed = a.speed
        pub.publish(cmd)
        rclpy.spin_once(n, timeout_sec=0.05)
        g = gap()
        min_gap = min(min_gap, g)
        max_v = max(max_v, st['v'])
        reasons.add(st['status'].get('reason'))
        if time.time() - last_print > 0.5:
            last_print = time.time()
            ss = st['status']
            print(f"    t={time.time() - t0:4.1f}s gap {g:6.2f} m  v {st['v']:4.1f}  shield {ss.get('reason')} "
                  f"obstacle {ss.get('static_distance')} v_static {ss.get('v_static')}")
        if g < -0.5:
            break                     # drove through the box position: no point continuing
        if max_v > 2.0 and abs(st['v']) < 0.05 and stopped_at is None and g > -0.5:
            stopped_at = g
        if stopped_at is not None and time.time() - t0 > 3 and abs(st['v']) < 0.05:
            break
    remove.publish(String(data='shield_box'))
    rclpy.spin_once(n, timeout_sec=0.2)

    ok_stop = stopped_at is not None and min_gap > 0
    ok_gap = a.min_gap <= min_gap <= a.max_gap
    print(f'  top speed reached {max_v:.1f} m/s, shield reasons seen: {sorted(r for r in reasons if r)}')
    print(f"  [{'PASS' if ok_stop else 'FAIL'}] car stopped before the box")
    print(f"  [{'PASS' if ok_gap else 'FAIL'}] closest gap {min_gap:.2f} m (allowed {a.min_gap}-{a.max_gap} m)")
    ok = ok_stop and ok_gap
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
