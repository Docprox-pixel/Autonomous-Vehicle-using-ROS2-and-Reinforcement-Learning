#!/usr/bin/env python3
"""Send a destination given in WORLD (map) coordinates, e.g. read off the Webots scene.

  python3 tools/send_goal.py 47.7 -20.0

The car's odom frame starts at its start pose with world-aligned axes, so the goal in odom is
world goal - start position (start position = GPS - 1 m forward: the GPS sits on the roof mount).
In RViz you can instead click "2D Goal Pose" (that goal is already in odom).
"""
import math
import sys
import time

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String


def main():
    gx, gy = float(sys.argv[1]), float(sys.argv[2])
    rclpy.init()
    n = Node('send_goal')
    st = {}
    n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m.point), 5)
    n.create_subscription(Imu, '/imu/data', lambda m: st.__setitem__('q', m.orientation), 5)
    n.create_subscription(Odometry, '/odometry/filtered', lambda m: st.__setitem__('odom', m.pose.pose.position), 5)
    n.create_subscription(String, '/route/status', lambda m: st.__setitem__('route', m.data), 5)
    pub = n.create_publisher(PoseStamped, '/goal_pose', 5)
    while not all(k in st for k in ('gps', 'q', 'odom')):
        rclpy.spin_once(n, timeout_sec=0.1)
    q = st['q']
    yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    # world position of base_link now, and odom position now -> offset world - odom
    bx, by = st['gps'].x - math.cos(yaw), st['gps'].y - math.sin(yaw)
    ox, oy = bx - st['odom'].x, by - st['odom'].y
    g = PoseStamped()
    g.header.frame_id = 'odom'
    g.pose.position.x, g.pose.position.y = gx - ox, gy - oy
    g.pose.orientation.w = 1.0
    t0 = time.time()
    st.pop('route', None)
    while time.time() - t0 < 15 and '"route"' not in st.get('route', ''):
        pub.publish(g)
        rclpy.spin_once(n, timeout_sec=0.3)
    print(f"goal world ({gx}, {gy}) -> odom ({g.pose.position.x:.1f}, {g.pose.position.y:.1f}): "
          f"{st.get('route', 'no route status (is route_planner running?)')}")
    n.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
