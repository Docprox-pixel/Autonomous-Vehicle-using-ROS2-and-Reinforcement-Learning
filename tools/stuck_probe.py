#!/usr/bin/env python3
"""When the car has been standing still for > N s, save what the planner sees (obstacles, road
walls, lane walls, route, camera) to OUT.npz/.png and exit.  python3 tools/stuck_probe.py SECONDS OUT"""
import json, math, sys, time
import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
rclpy.init(); n = Node('stuck_probe'); st = {}
q = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
for t in ('/perception/obstacle_points', '/perception/road_boundary'):
    n.create_subscription(PointCloud2, t, lambda m, t=t: st.__setitem__(t, point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))), 5)
n.create_subscription(PointCloud2, '/route/lane_walls', lambda m: st.__setitem__('walls', point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))), q)
n.create_subscription(Path, '/route/path', lambda m: st.__setitem__('path', np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])), q)
n.create_subscription(Odometry, '/odometry/filtered', lambda m: st.__setitem__('odom', m), 10)
n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m.point), 5)
n.create_subscription(Image, '/camera/image_raw', lambda m: st.__setitem__('img', m), 1)
n.create_subscription(String, '/planner/status', lambda m: st.__setitem__('pl', m.data), 5)
limit, out = float(sys.argv[1]), sys.argv[2]
still = None; t0 = time.time()
while time.time() - t0 < 1200:
    rclpy.spin_once(n, timeout_sec=0.05)
    if 'odom' not in st:
        continue
    v = abs(st['odom'].twist.twist.linear.x)
    still = (still or time.time()) if v < 0.05 else None
    if still and time.time() - still > limit and all(k in st for k in ('/perception/obstacle_points', 'gps', 'img')):
        o = st['odom'].pose.pose
        yaw = math.atan2(2 * o.orientation.w * o.orientation.z, 1 - 2 * o.orientation.z ** 2)
        np.savez(out + '.npz', obs=st['/perception/obstacle_points'], road=st.get('/perception/road_boundary', np.zeros((0, 3))),
                 walls=st.get('walls', np.zeros((0, 3))), path=st.get('path', np.zeros((0, 2))),
                 pose=[o.position.x, o.position.y, yaw], gps=[st['gps'].x, st['gps'].y])
        from PIL import Image as PI
        m = st['img']; PI.fromarray(np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 4)[:, :, [2, 1, 0]]).save(out + '.png')
        print('stuck at', round(st['gps'].x, 1), round(st['gps'].y, 1), 'planner', st.get('pl', '')[:200])
        break
