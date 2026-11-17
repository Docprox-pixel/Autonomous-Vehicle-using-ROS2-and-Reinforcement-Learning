#!/usr/bin/env python3
"""Live blind-spot test: put small boxes right next to the car and check the lidars see them.

Uses the Ros2Supervisor (launched by sim.launch.py) to insert 0.3 x 0.3 x 0.5 m boxes at fixed
positions in the CAR frame, then counts /lidar/points (base_link, all lidars merged) inside each
box. Simulator-agnostic except for the spawning service.

  python3 test/blindspot_live_test.py            # sim running, car parked
"""
import math
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from webots_ros2_msgs.srv import SpawnNodeFromString

GROUND_Z = -0.32       # road height in base_link (config/sensor_layout.yaml)
GPS_X = 1.0            # GPS is on the roof mount, 1.0 m ahead of base_link
SIZE = (0.3, 0.3, 0.5)
# (name, x, y) in the car frame. Car body spans x -1.15..4.05, y -1.0..1.0.
PROBES = [
    ('probe_left_door', 1.5, 2.0),     # 1 m beside the left door
    ('probe_right_door', 1.5, -2.0),   # 1 m beside the right door
    ('probe_front', 5.05, 0.0),        # 1 m ahead of the bumper
    ('probe_rear', -2.15, 0.0),        # 1 m behind the bumper
    ('probe_front_left', 4.8, 1.8),    # diagonal, front-left corner
    ('probe_rear_right', -1.9, -1.8),  # diagonal, rear-right corner
]


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class BlindSpotTest(Node):
    def __init__(self):
        super().__init__('blindspot_live_test')
        self.gps = self.imu = self.cloud = None
        self.create_subscription(PointStamped, '/gps/position', lambda m: setattr(self, 'gps', m), 10)
        self.create_subscription(Imu, '/imu/data', lambda m: setattr(self, 'imu', m), 10)
        self.create_subscription(PointCloud2, '/lidar/points', lambda m: setattr(self, 'cloud', m),
                                 qos_profile_sensor_data)
        self.spawn = self.create_client(SpawnNodeFromString, '/Ros2Supervisor/spawn_node_from_string')
        self.remove = self.create_publisher(String, '/Ros2Supervisor/remove_node', 10)

    def spin_for(self, s):
        end = time.time() + s
        while time.time() < end:
            rclpy.spin_once(self, timeout_sec=0.05)

    def fresh_cloud(self):
        self.cloud = None
        while self.cloud is None:
            rclpy.spin_once(self, timeout_sec=0.05)
        return point_cloud2.read_points_numpy(self.cloud, field_names=('x', 'y', 'z'))

    def count_in(self, pts, x, y):
        hx, hy = SIZE[0] / 2 + 0.15, SIZE[1] / 2 + 0.15
        m = ((np.abs(pts[:, 0] - x) < hx) & (np.abs(pts[:, 1] - y) < hy) &
             (pts[:, 2] > GROUND_Z + 0.05) & (pts[:, 2] < GROUND_Z + SIZE[2] + 0.05))
        return int(m.sum())

    def run(self):
        end = time.time() + 30
        while time.time() < end and (self.gps is None or self.imu is None or self.cloud is None):
            rclpy.spin_once(self, timeout_sec=0.1)
        if self.gps is None or self.cloud is None:
            print('FAIL: no /gps/position or /lidar/points')
            return False
        if not self.spawn.wait_for_service(timeout_sec=10):
            print('FAIL: /Ros2Supervisor/spawn_node_from_string not available')
            return False

        before = self.fresh_cloud()
        yaw = yaw_of(self.imu.orientation)
        c, s = math.cos(yaw), math.sin(yaw)
        bx, by = self.gps.point.x - GPS_X * c, self.gps.point.y - GPS_X * s
        ground_world = self.gps.point.z - (1.95 - GROUND_Z)   # roof mount z=1.95 in base_link

        for name, x, y in PROBES:
            wx, wy = bx + c * x - s * y, by + s * x + c * y
            node = (f'Solid {{ name "{name}" translation {wx:.3f} {wy:.3f} {ground_world + SIZE[2] / 2:.3f} '
                    f'rotation 0 0 1 {yaw:.4f} children [ Shape {{ appearance PBRAppearance {{ baseColor 1 0.4 0 }} '
                    f'geometry Box {{ size {SIZE[0]} {SIZE[1]} {SIZE[2]} }} }} ] }}')
            fut = self.spawn.call_async(SpawnNodeFromString.Request(data=node))
            rclpy.spin_until_future_complete(self, fut, timeout_sec=5)
            if not (fut.result() and fut.result().success):
                print(f'FAIL: could not spawn {name}')
                return False

        self.spin_for(1.0)
        after = self.fresh_cloud()
        print(f'{len(after)} points in merged cloud. Boxes 0.5 m tall, 1 m from the car body:')
        results = []
        for name, x, y in PROBES:
            n0, n1 = self.count_in(before, x, y), self.count_in(after, x, y)
            ok = n1 - n0 >= 5
            results.append(ok)
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<17} at ({x:+.2f},{y:+.2f}) m: {n1 - n0:4d} lidar points")

        for name, _, _ in PROBES:
            self.remove.publish(String(data=name))
        self.spin_for(0.5)

        box = ((after[:, 0] > -1.15) & (after[:, 0] < 4.05) & (np.abs(after[:, 1]) < 1.0) &
               (after[:, 2] > GROUND_Z + 0.05) & (after[:, 2] < 1.45))
        print(f'  [INFO] self-hits (points on the car body): {int(box.sum())}'
              f' -> filtered by the perception pipeline')
        ok = all(results)
        print(f"\nRESULT: {'ALL PASS' if ok else 'FAILURES'} ({sum(results)}/{len(results)})")
        return ok


def main():
    rclpy.init()
    node = BlindSpotTest()
    node.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])
    try:
        ok = node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
