#!/usr/bin/env python3
"""Goal-directed turn at a junction: does the car turn left/right when the goal says so?

The car starts in city.wbt at world (39.66, 0.03) heading south (-y) and reaches the 4-way
junction at (45, -45) after ~45 m. A goal beyond the junction to the EAST needs a LEFT turn,
to the WEST a RIGHT turn. The odom frame starts at the car origin with world-aligned axes (the
EKF takes its heading from the IMU), so odom = world - start.

  python3 test/junction_goal_test.py --turn left      (launch with stack:=false: the test
  python3 test/junction_goal_test.py --turn right      starts the planner AFTER sending the goal)

If the planner is already driving when the goal arrives, the car may be past the junction by
then (it drives off at 8 m/s during start-up), which made an earlier version of this test fail.
"""
import argparse
import math
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu

START = (39.66, 0.03)
GOALS = {'left': (72.0, -51.0), 'right': (0.0, -41.0)}   # world coordinates, on the target lanes
# Correct lane after the turn (driving on the right). The cross street runs along y = -45:
#   right turn -> heading west on the west arm (x < 33): our side is the NORTH half, y > -45
#   left turn  -> heading east on the east arm (x > 57): our side is the SOUTH half, y < -45
LANE = {'right': (lambda x, y: 10 < x < 33 and abs(y + 45) < 12, lambda y: y > -45.0),
        'left': (lambda x, y: 57 < x < 75 and abs(y + 45) < 12, lambda y: y < -45.0)}


def np_mean(v):
    return sum(v) / len(v) if v else 0.0


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--turn', choices=GOALS, required=True)
    ap.add_argument('--seconds', type=float, default=60)
    a = ap.parse_args()
    rclpy.init()
    n = Node('junction_goal_test')
    st = {'gps': None, 'yaw': None}
    n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m.point), 10)
    n.create_subscription(Imu, '/imu/data', lambda m: st.__setitem__('yaw', yaw_of(m.orientation)), 10)
    pub = n.create_publisher(PoseStamped, '/goal_pose', 10)
    while st['gps'] is None or st['yaw'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    gx, gy = GOALS[a.turn]
    goal = PoseStamped()
    goal.header.frame_id = 'odom'
    goal.pose.position.x, goal.pose.position.y = gx - START[0], gy - START[1]
    goal.pose.orientation.w = 1.0
    for _ in range(5):
        pub.publish(goal)
        rclpy.spin_once(n, timeout_sec=0.1)
    print(f'goal {a.turn}: world ({gx}, {gy}) -> odom ({goal.pose.position.x:.1f}, {goal.pose.position.y:.1f})')
    planner = subprocess.Popen(['ros2', 'run', 'auto_car', 'local_planner.py', '--ros-args', '-p', 'use_sim_time:=true'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    yaw0 = st['yaw']
    t0, last = time.time(), time.time()
    best = math.inf
    on_arm = []                        # y samples once the car is on the target road
    while time.time() - t0 < a.seconds:
        if time.time() - t0 < 3.0:
            pub.publish(goal)         # until the planners have subscribed (each goal = a re-plan)
        rclpy.spin_once(n, timeout_sec=0.05)
        g = st['gps']
        best = min(best, math.hypot(g.x - gx, g.y - gy))
        if LANE[a.turn][0](g.x, g.y):
            on_arm.append(g.y)
        if time.time() - last > 5:
            last = time.time()
            print(f'  t={time.time() - t0:3.0f}s at ({g.x:6.1f},{g.y:6.1f}) heading change '
                  f'{math.degrees(math.atan2(math.sin(st["yaw"] - yaw0), math.cos(st["yaw"] - yaw0))):+5.0f} deg, '
                  f'{math.hypot(g.x - gx, g.y - gy):5.1f} m to goal')
        if best < 6.0:
            break
    planner.terminate()
    turn = math.degrees(math.atan2(math.sin(st['yaw'] - yaw0), math.cos(st['yaw'] - yaw0)))
    want = 1 if a.turn == 'left' else -1
    ok_turn = turn * want > 60
    ok_goal = best < 10.0
    print(f"  [{'PASS' if ok_turn else 'FAIL'}] turned {a.turn}: heading change {turn:+.0f} deg")
    print(f"  [{'PASS' if ok_goal else 'FAIL'}] reached the goal area: closest {best:.1f} m")
    right_side = [LANE[a.turn][1](y) for y in on_arm]
    ok_lane = bool(on_arm) and np_mean(right_side) > 0.9
    print(f"  [{'PASS' if ok_lane else 'FAIL'}] own side of the new road: {100 * np_mean(right_side) if on_arm else 0:.0f}% "
          f"of {len(on_arm)} samples (y median {sorted(on_arm)[len(on_arm) // 2] if on_arm else float('nan'):.1f})")
    ok_turn = ok_turn and ok_lane
    print(f"\nRESULT: {'PASS' if ok_turn and ok_goal else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok_turn and ok_goal else 1)


if __name__ == '__main__':
    main()
