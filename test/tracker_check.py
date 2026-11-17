#!/usr/bin/env python3
"""Check the object tracker.

  python3 test/tracker_check.py --expect-static --drive   # static world: 0 moving while ego drives
  python3 test/tracker_check.py --expect-moving           # traffic world: moving cars found & removed

Measures for --seconds:
  * unique moving track IDs and their speeds
  * whether moving objects are absent from /perception/static_points (map input)
"""
import argparse
import sys
import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from auto_car.msg import TrackArray
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class TrackerCheck(Node):
    def __init__(self):
        super().__init__('tracker_check')
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.tracks = None
        self.static = None
        self.create_subscription(TrackArray, '/perception/tracks', self.on_tracks, 10)
        self.statics = {}        # stamp -> static cloud, to compare with tracks of the SAME frame
        self.create_subscription(PointCloud2, '/perception/static_points', self.on_static, 5)
        self.pub = self.create_publisher(AckermannDriveStamped, '/cmd_drive', 10)
        self.moving = {}         # id -> list of speeds
        self.confirmed = set()
        self.leaks = []          # moving tracks still present in the static cloud
        self.frames = 0

    def on_tracks(self, msg):
        self.tracks = msg
        self.frames += 1
        for t in msg.tracks:
            self.confirmed.add(t.id)
            if t.moving:
                self.moving.setdefault(t.id, []).append(t.speed)

    def on_static(self, m):
        key = (m.header.stamp.sec, m.header.stamp.nanosec)
        self.statics[key] = m
        if len(self.statics) > 20:
            self.statics.pop(next(iter(self.statics)))

    def leak_check(self):
        if self.tracks is None:
            return
        key = (self.tracks.header.stamp.sec, self.tracks.header.stamp.nanosec)
        static = self.statics.get(key)
        if static is None:
            return
        pts = point_cloud2.read_points_numpy(static, field_names=('x', 'y', 'z'))
        for t in self.tracks.tracks:
            if not t.moving:
                continue
            hx, hy = t.size.x / 2, t.size.y / 2     # the object's own footprint (not neighbours)
            inside = ((np.abs(pts[:, 0] - t.position.x) < hx) & (np.abs(pts[:, 1] - t.position.y) < hy)).sum()
            self.leaks.append(int(inside))

    def run(self, seconds, drive):
        t0 = time.time()
        last_leak = 0
        while time.time() - t0 < seconds:
            if drive:
                el = time.time() - t0
                cmd = AckermannDriveStamped()
                cmd.drive.speed = 4.0 if el < seconds - 3 else 0.0
                cmd.drive.steering_angle = 0.0 if el < 8 else 0.2
                self.pub.publish(cmd)
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.time() - last_leak > 1.0:
                self.leak_check()
                last_leak = time.time()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=20.0)
    ap.add_argument('--drive', action='store_true')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--expect-static', action='store_true')
    g.add_argument('--expect-moving', action='store_true')
    a = ap.parse_args()

    rclpy.init()
    n = TrackerCheck()
    n.run(a.seconds, a.drive)
    print(f'{n.frames} tracker frames, {len(n.confirmed)} confirmed objects, {len(n.moving)} moving')
    for tid, sp in sorted(n.moving.items()):
        print(f'  moving #{tid}: {len(sp)} frames, speed median {np.median(sp):.1f} m/s, max {max(sp):.1f}')
    ok = n.frames > 50
    if a.expect_static:
        ok &= len(n.moving) == 0
        print(f"[{'PASS' if len(n.moving) == 0 else 'FAIL'}] no false 'moving' objects in a static world"
              f"{' while driving' if a.drive else ''}")
    else:
        plausible = all(1.0 < np.median(sp) < 25.0 for sp in n.moving.values())
        ok &= len(n.moving) >= 1 and plausible
        print(f"[{'PASS' if n.moving else 'FAIL'}] moving vehicles detected: {len(n.moving)}")
        print(f"[{'PASS' if plausible and n.moving else 'FAIL'}] speeds plausible (1-25 m/s)")
        leak = float(np.mean(n.leaks)) if n.leaks else float('nan')
        leak_ok = n.leaks and leak < 5
        ok &= bool(leak_ok)
        print(f"[{'PASS' if leak_ok else 'FAIL'}] moving objects removed from map input: "
              f"{leak:.1f} leftover points per moving object (checks: {len(n.leaks)})")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
