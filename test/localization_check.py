#!/usr/bin/env python3
"""Phase 1 check: how accurate are the EKF and SLAM poses, without using GPS?

GPS is used ONLY here, as ground truth. The localisation itself never sees it.

Frames: the EKF takes its heading from the IMU (absolute yaw), so the odom frame's axes
are aligned with the world, and odom/map start where the car starts. Ground truth is
therefore (GPS - GPS_at_start), corrected for the GPS sitting `--gps-x` m ahead of base_link.

Usage (sim running with slam:=true):  python3 test/localization_check.py
"""
import argparse
import math
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Imu
from tf2_ros import Buffer, TransformListener


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class LocCheck(Node):
    def __init__(self, gps_x):
        super().__init__('localization_check')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.gps_x = gps_x
        self.gps = self.imu = self.ekf = self.map = None
        self.create_subscription(PointStamped, '/gps/position', lambda m: setattr(self, 'gps', m), 10)
        self.create_subscription(Imu, '/imu/data', lambda m: setattr(self, 'imu', m), 20)
        self.create_subscription(Odometry, '/odometry/filtered', lambda m: setattr(self, 'ekf', m), 20)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', lambda m: setattr(self, 'map', m), latched)
        self.pub = self.create_publisher(AckermannDriveStamped, '/cmd_drive', 10)
        self.tf = Buffer()
        self.tfl = TransformListener(self.tf, self)
        self.results = []

    def drive(self, seconds, speed=None, steer=0.0):
        end = time.time() + seconds
        while time.time() < end:
            if speed is not None:
                c = AckermannDriveStamped()
                c.drive.speed, c.drive.steering_angle = float(speed), float(steer)
                self.pub.publish(c)
            rclpy.spin_once(self, timeout_sec=0.05)

    def truth(self):
        """World pose of base_link from GPS (+ IMU heading for the lever arm)."""
        yaw = yaw_of(self.imu.orientation)
        return (self.gps.point.x - self.gps_x * math.cos(yaw),
                self.gps.point.y - self.gps_x * math.sin(yaw), yaw)

    def slam_pose(self):
        try:
            t = self.tf.lookup_transform('map', 'base_link', Time())
        except Exception:
            return None
        return t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation)

    def check(self, name, ok, detail):
        self.results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<13} {detail}")

    def run(self):
        end = time.time() + 30
        while time.time() < end and (self.gps is None or self.imu is None or self.ekf is None
                                     or self.slam_pose() is None):
            rclpy.spin_once(self, timeout_sec=0.1)
        missing = [n for n, ok in (('/gps/position', self.gps), ('/imu/data', self.imu),
                                   ('/odometry/filtered (EKF)', self.ekf),
                                   ('TF map->base_link (SLAM)', self.slam_pose())) if ok is None]
        if missing:
            print(f'FAIL: nothing on {missing}')
            return False

        self.drive(1.0)                              # settle
        x0, y0, _ = self.truth()
        e0 = self.ekf.pose.pose.position
        s0 = self.slam_pose()
        print(f'Start: truth=({x0:.2f},{y0:.2f})  EKF=({e0.x:.2f},{e0.y:.2f})  SLAM=({s0[0]:.2f},{s0[1]:.2f})')
        print('Driving: straight ~25 m, then a left turn...')
        self.drive(7.0, speed=4.0)
        self.drive(4.0, speed=4.0, steer=0.25)
        self.drive(4.0, speed=0.0)                   # stop and let SLAM catch up
        self.drive(2.5)

        tx, ty, tyaw = self.truth()
        dx, dy = tx - x0, ty - y0
        dist_true = math.hypot(dx, dy)
        e = self.ekf.pose.pose
        ex, ey, eyaw = e.position.x - e0.x, e.position.y - e0.y, yaw_of(e.orientation)
        s = self.slam_pose()
        sx, sy, syaw = s[0] - s0[0], s[1] - s0[1], s[2]

        ekf_err = math.hypot(ex - dx, ey - dy)
        slam_err = math.hypot(sx - dx, sy - dy)
        print(f'Travelled {dist_true:.1f} m (GPS truth), heading now {math.degrees(tyaw):+.1f} deg')
        self.check('EKF position', ekf_err < 0.03 * dist_true + 0.3,
                   f'error {ekf_err:.2f} m ({100 * ekf_err / max(dist_true, 1e-6):.1f}% of distance)')
        self.check('EKF heading', abs(math.degrees(wrap(eyaw - tyaw))) < 3,
                   f'error {math.degrees(wrap(eyaw - tyaw)):+.2f} deg')
        self.check('SLAM position', slam_err < 1.0,
                   f'error {slam_err:.2f} m ({100 * slam_err / max(dist_true, 1e-6):.1f}% of distance)')
        self.check('SLAM heading', abs(math.degrees(wrap(syaw - tyaw))) < 3,
                   f'error {math.degrees(wrap(syaw - tyaw)):+.2f} deg')

        if self.map is None:
            self.check('map', False, 'no /map received')
        else:
            g = np.asarray(self.map.data)
            occ, free = int((g > 50).sum()), int((g == 0).sum())
            area = free * self.map.info.resolution ** 2
            self.check('map', occ > 200 and area > 500,
                       f'{self.map.info.width}x{self.map.info.height} cells @ {self.map.info.resolution} m: '
                       f'{occ} occupied, {area:.0f} m^2 free')

        ok = all(self.results)
        print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES'} ({sum(self.results)}/{len(self.results)})")
        return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gps-x', type=float, default=1.0, help='GPS position ahead of base_link (m)')
    args = ap.parse_args()
    rclpy.init()
    node = LocCheck(args.gps_x)
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
