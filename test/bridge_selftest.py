#!/usr/bin/env python3
"""Drive the car and check that a simulator bridge honours the sensor contract.

Simulator-independent: works for any bridge (Webots now, Isaac Sim later).
Ground truth comes from sensors the bridge does not compute itself:
  * IMU orientation  -> true heading / turn direction
  * GPS position     -> true distance travelled / direction of motion

Checks:
  1. speed      : /wheel/odom speed  ~= GPS distance / time          (km/h vs m/s bug)
  2. forward    : GPS motion direction ~= IMU heading                 (car drives forward)
  3. left turn  : +steering makes IMU yaw INCREASE (CCW = left)       (steering sign)
  4. odom yaw   : /wheel/odom yaw change has same sign/size as IMU
  5. lidar front: obstacles at scan angle 0 get closer when driving   (scan orientation)
  6. stop       : speed returns to ~0 after commands stop             (watchdog)

Usage (simulation running):  python3 test/bridge_selftest.py
"""
import math
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


class SelfTest(Node):
    def __init__(self):
        super().__init__('bridge_selftest', parameter_overrides=[])
        self.imu = self.odom = self.gps = self.scan = None
        self.create_subscription(Imu, '/imu/data', lambda m: setattr(self, 'imu', m), 20)
        self.create_subscription(Odometry, '/wheel/odom', lambda m: setattr(self, 'odom', m), 20)
        self.create_subscription(PointStamped, '/gps/position', lambda m: setattr(self, 'gps', m), 10)
        self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m), qos_profile_sensor_data)
        self.pub = self.create_publisher(AckermannDriveStamped, '/cmd_drive', 10)
        self.results = []

    # ------------------------------------------------------------ helpers
    def spin_for(self, seconds, speed=None, steer=0.0):
        end = time.time() + seconds
        while time.time() < end:
            if speed is not None:
                cmd = AckermannDriveStamped()
                cmd.drive.speed = float(speed)
                cmd.drive.steering_angle = float(steer)
                self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for_data(self, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.1)
            if all(x is not None for x in (self.imu, self.odom, self.gps, self.scan)):
                return True
        missing = [n for n, x in (('imu', self.imu), ('odom', self.odom), ('gps', self.gps), ('scan', self.scan))
                   if x is None]
        print(f'FAIL: no data on {missing}')
        return False

    def snapshot(self):
        g = self.gps
        return {'t': stamp(g), 'x': g.point.x, 'y': g.point.y,
                'imu_yaw': yaw_of(self.imu.orientation), 'odom_yaw': yaw_of(self.odom.pose.pose.orientation),
                'speed': self.odom.twist.twist.linear.x, 'front': self.front_range(), 'scan': self.scan}

    def front_range(self, half_angle=math.radians(10)):
        s = self.scan
        a = s.angle_min + np.arange(len(s.ranges)) * s.angle_increment
        r = np.asarray(s.ranges)
        m = (np.abs(np.arctan2(np.sin(a), np.cos(a))) < half_angle) & np.isfinite(r)
        return float(np.median(r[m])) if m.any() else float('nan')

    def next_scan(self, **drive):
        """Block until a new /scan arrives (keeps driving); return (angles, ranges, imu_yaw, gps_xy)."""
        old = stamp(self.scan)
        while stamp(self.scan) == old:
            self.spin_for(0.02, **drive)
        s = self.scan
        a = s.angle_min + np.arange(len(s.ranges)) * s.angle_increment
        return a, np.asarray(s.ranges, dtype=float), yaw_of(self.imu.orientation), (self.gps.point.x, self.gps.point.y)

    def lidar_forward_slope(self, speed=4.0):
        """Fit dr = k*cos(angle) between two scans while driving straight. Expect k ~ -distance."""
        a, r0, _, p0 = self.next_scan(speed=speed)
        for _ in range(6):                      # ~0.6 s later
            a, r1, _, p1 = self.next_scan(speed=speed)
        d = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        with np.errstate(invalid="ignore"):   # inf - inf for no-return rays (masked below)
            dr = r1 - r0
        m = np.isfinite(r0) & np.isfinite(r1) & (np.abs(dr) < 2 * d + 0.5)
        c = np.cos(a[m])
        k = float(np.dot(c, dr[m]) / np.dot(c, c))
        return k, d, int(m.sum())

    def lidar_rotation_shift(self, speed=4.0, steer=0.25):
        """Best circular shift (rad) aligning two scans during a left turn. Expect ~ -IMU yaw change."""
        a, r0, y0, _ = self.next_scan(speed=speed, steer=steer)
        for _ in range(5):
            a, r1, y1, _ = self.next_scan(speed=speed, steer=steer)
        inc = a[1] - a[0]
        dyaw = wrap(y1 - y0)
        f0, f1 = np.nan_to_num(r0, posinf=100.0), np.nan_to_num(r1, posinf=100.0)
        max_shift = int(abs(dyaw) / inc * 3) + 5
        errs = {s: np.median(np.abs(f1 - np.roll(f0, s))) for s in range(-max_shift, max_shift + 1)}
        best = min(errs, key=errs.get)
        # np.roll(f0, s)[i] = f0[i - s]: object seen at angle a now appears at a + s*inc.
        return best * inc, dyaw

    def check(self, name, ok, detail):
        self.results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:<11} {detail}")

    # --------------------------------------------------------------- test
    def run(self):
        if not self.wait_for_data():
            return False
        print('Data on all contract topics. Driving...')

        # Phase A: straight, 4 m/s
        self.spin_for(3.0, speed=4.0)                 # accelerate
        a0 = self.snapshot()
        speeds = []
        end = time.time() + 4.0
        while time.time() < end:
            self.spin_for(0.1, speed=4.0)
            speeds.append(self.odom.twist.twist.linear.x)
        a1 = self.snapshot()
        dt = a1['t'] - a0['t']
        dist = math.hypot(a1['x'] - a0['x'], a1['y'] - a0['y'])
        gps_speed = dist / dt if dt > 0 else float('nan')
        odom_speed = float(np.mean(speeds))
        self.check('speed', abs(gps_speed - odom_speed) < 0.15 * max(gps_speed, 0.5) and gps_speed > 1.0,
                   f'odom {odom_speed:.2f} m/s vs GPS {gps_speed:.2f} m/s (cmd 4.0)')

        motion_dir = math.atan2(a1['y'] - a0['y'], a1['x'] - a0['x'])
        heading_err = math.degrees(abs(wrap(motion_dir - a1['imu_yaw'])))
        self.check('forward', heading_err < 15, f'GPS motion direction vs IMU heading: {heading_err:.1f} deg')

        k, d, n = self.lidar_forward_slope()
        self.check('lidar front', k < -0.5 * d,
                   f'range change fit dr = {k:+.2f}*cos(angle) after moving {d:.2f} m '
                   f'(expect ~{-d:.2f}; positive = scan 0 deg points backwards) [{n} rays]')

        # Phase B: left turn, steering +0.25 rad
        shift, dyaw = self.lidar_rotation_shift()
        self.check('lidar ccw', shift * dyaw < 0 and abs(abs(shift) - abs(dyaw)) < 0.5 * abs(dyaw) + 0.02,
                   f'scene shifted {math.degrees(shift):+.1f} deg while car yawed {math.degrees(dyaw):+.1f} deg '
                   f'(expect opposite sign; same sign = left/right mirrored)')
        b0 = self.snapshot()
        self.spin_for(4.0, speed=4.0, steer=0.25)
        b1 = self.snapshot()
        d_imu = math.degrees(wrap(b1['imu_yaw'] - b0['imu_yaw']))
        d_odom = math.degrees(wrap(b1['odom_yaw'] - b0['odom_yaw']))
        self.check('left turn', d_imu > 5, f'steer +0.25 rad -> IMU yaw change {d_imu:+.1f} deg (+ = left)')
        self.check('odom yaw', d_odom * d_imu > 0 and abs(d_odom - d_imu) < 0.3 * abs(d_imu) + 2,
                   f'odom {d_odom:+.1f} deg vs IMU {d_imu:+.1f} deg')

        # Phase C: stop commanding -> watchdog
        self.spin_for(6.0, speed=None)
        v = self.odom.twist.twist.linear.x
        self.check('stop', abs(v) < 0.3, f'speed {v:.2f} m/s after commands stopped')

        ok = all(self.results)
        print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES'} ({sum(self.results)}/{len(self.results)})")
        return ok


def main():
    rclpy.init()
    node = SelfTest()
    node.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
