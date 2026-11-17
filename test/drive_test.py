#!/usr/bin/env python3
"""Autonomous drive test: the local planner drives, this script only watches.

  python3 test/drive_test.py --seconds 90
  python3 test/drive_test.py --goal X Y --seconds 150     (world coordinates; launch with stack:=false,
                                                          the test sends the goal, then starts the planner)

Ground truth from GPS (distance, turns) and the obstacle cloud (clearance to the car body).
PASS if: distance >= --min-distance, turns >= --min-turns, min clearance >= --min-clearance,
never stuck longer than --max-stuck seconds.
"""
import argparse
import json
import math
import subprocess
import sys
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String

BOX = {'x': (-1.15, 4.05), 'y': (-1.0, 1.0)}   # car body in base_link
Z_BAND = (-0.07, 1.9)


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def body_clearance(pts, with_point=False):
    p = pts[(pts[:, 2] > Z_BAND[0]) & (pts[:, 2] < Z_BAND[1])]
    if len(p) == 0:
        return (math.inf, None) if with_point else math.inf
    dx = np.maximum.reduce([BOX['x'][0] - p[:, 0], np.zeros(len(p)), p[:, 0] - BOX['x'][1]])
    dy = np.maximum.reduce([BOX['y'][0] - p[:, 1], np.zeros(len(p)), p[:, 1] - BOX['y'][1]])
    d = np.hypot(dx, dy)
    keep = d > 0.05                     # ignore anything already "inside" (self-hits, noise)
    if not keep.any():
        return (math.inf, None) if with_point else math.inf
    # Same rule as the shield's contact check: a real obstacle returns several points, so the
    # clearance is the distance at which at least TWO points are found. Single stray returns at the
    # body edge (near the corner lidar mounts, ~0.2 m from the sensor) do not count.
    dk = np.where(keep, d, np.inf)
    if np.isfinite(dk).sum() < 2:
        return (math.inf, None) if with_point else math.inf
    i = int(np.argsort(dk)[1])
    return (float(d[i]), p[i]) if with_point else float(d[i])


def send_goal_until_routed(node, pub, goal, timeout=15.0):
    """Publish the goal until the route planner confirms a route (discovery can drop early messages)."""
    import json as _json
    from std_msgs.msg import String as _S
    got = {}
    sub = node.create_subscription(_S, '/route/status', lambda m: got.__setitem__('s', _json.loads(m.data)), 10)
    t0, last = time.time(), 0.0
    while time.time() - t0 < timeout and got.get('s', {}).get('state') != 'route':
        if time.time() - last > 0.5:
            pub.publish(goal)
            last = time.time()
        rclpy.spin_once(node, timeout_sec=0.05)
    node.destroy_subscription(sub)
    ok = got.get('s', {}).get('state') == 'route'
    print(f"route planner: {'route ready' if ok else 'NO ROUTE (goal not confirmed)'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seconds', type=float, default=90)
    ap.add_argument('--min-distance', type=float, default=250)
    ap.add_argument('--min-turns', type=int, default=2)
    ap.add_argument('--min-clearance', type=float, default=0.5)
    ap.add_argument('--max-stuck', type=float, default=10)
    ap.add_argument('--goal', type=float, nargs=2, help='world x y: drive there via the lane map')
    ap.add_argument('--goal-tolerance', type=float, default=8.0)
    ap.add_argument('--planner', default='arcs', choices=['arcs', 'rl'], help="'rl' = trained SAC policy node")
    ap.add_argument('--model', default='', help='SAC model .zip for --planner rl')
    a = ap.parse_args()

    rclpy.init()
    n = Node('drive_test')
    st = {'gps': None, 'yaw': None, 'clear': [], 'shield': [], 'planner': {}}
    n.create_subscription(PointStamped, '/gps/position', lambda m: st.__setitem__('gps', m.point), 10)
    n.create_subscription(Imu, '/imu/data', lambda m: st.__setitem__('yaw', yaw_of(m.orientation)), 10)
    def on_obs(m):
        c_now, pt = body_clearance(point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z')), True)
        # Confirmed over two consecutive frames: a real obstacle is still there 0.1 s later; one-frame
        # phantom returns (the car then drove through the spot without contact) are not approaches.
        c = max(c_now, st.get('prev_clear', math.inf))
        st['prev_clear'] = c_now
        st['clear'].append(c)
        if c < st.get('min_clear', (math.inf,))[0] and st['gps'] is not None and st['yaw'] is not None:
            yaw = st['yaw']
            bx, by = st['gps'].x - math.cos(yaw), st['gps'].y - math.sin(yaw)
            wx = bx + math.cos(yaw) * pt[0] - math.sin(yaw) * pt[1]
            wy = by + math.sin(yaw) * pt[0] + math.cos(yaw) * pt[1]
            st['min_clear'] = (c, time.time(), st['gps'].x, st['gps'].y, dict(st['planner']),
                               (round(float(pt[0]), 2), round(float(pt[1]), 2), round(float(pt[2]) + 0.32, 2)), (round(wx, 1), round(wy, 1)))
    n.create_subscription(PointCloud2, '/perception/obstacle_points', on_obs, 5)
    n.create_subscription(String, '/safety/status', lambda m: st['shield'].append(json.loads(m.data).get('reason')), 10)
    n.create_subscription(String, '/planner/status', lambda m: st.__setitem__('planner', json.loads(m.data)), 10)
    st['rule'] = 'go'
    st['vmax'] = 0.0
    from nav_msgs.msg import Odometry as _Odom
    n.create_subscription(_Odom, '/wheel/odom', lambda m: st.__setitem__('vmax', max(st['vmax'], m.twist.twist.linear.x)), 10)
    n.create_subscription(String, '/behaviour/state', lambda m: st.__setitem__('rule', json.loads(m.data).get('reason')), 10)
    st['lane'] = []   # per frame: (off-road fraction under the car, centre line under the car)

    def on_bev(m):
        # road_perception's bird's-eye map: 0.5 m cells, car rear axle at the bottom centre
        bev = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width)
        n_ = m.width
        rows = slice(n_ - 1 - 8, n_ - 1)            # 0..4 m ahead of the rear axle (body)
        cols = slice(n_ // 2 - 2, n_ // 2 + 2)      # +/- 1 m
        under = bev[rows, cols]
        known = under[under > 0]
        if len(known) > 10:
            st['lane'].append((float(np.mean(known == 200)), bool((known == 255).any())))
    n.create_subscription(Image, '/perception/road_bev', on_bev, 2)

    while st['gps'] is None or st['yaw'] is None:
        rclpy.spin_once(n, timeout_sec=0.1)
    planner = None
    if a.goal:
        # odom = world - car origin at start (world-aligned axes); car origin = GPS - 1 m forward
        ox = st['gps'].x - 1.0 * math.cos(st['yaw'])
        oy = st['gps'].y - 1.0 * math.sin(st['yaw'])
        gp = PoseStamped()
        gp.header.frame_id = 'odom'
        gp.pose.position.x, gp.pose.position.y = a.goal[0] - ox, a.goal[1] - oy
        gp.pose.orientation.w = 1.0
        gpub = n.create_publisher(PoseStamped, '/goal_pose', 10)
        send_goal_until_routed(n, gpub, gp)
        exe = 'rl_planner.py' if a.planner == 'rl' else 'local_planner.py'
        extra = ['-p', f'model:={a.model}'] if a.planner == 'rl' and a.model else []
        planner = subprocess.Popen(['ros2', 'run', 'auto_car', exe, '--ros-args', '-p',
                                    'use_sim_time:=true'] + extra, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f'goal world ({a.goal[0]}, {a.goal[1]}) sent; planner started')
    goal_time = None
    dist, last = 0.0, (st['gps'].x, st['gps'].y)
    heading_acc, last_yaw, turns, turn_start = 0.0, st['yaw'], 0, st['yaw']
    stuck, max_stuck, last_move, stuck_at = 0.0, 0.0, time.time(), None
    path = [last]
    t0 = tp = time.time()
    while time.time() - t0 < a.seconds:
        rclpy.spin_once(n, timeout_sec=0.05)
        g = st['gps']
        step = math.hypot(g.x - last[0], g.y - last[1])
        if step > 0.5:
            dist += step
            last = (g.x, g.y)
            path.append(last)
            last_move = time.time()
        elif st['rule'] in ('red_light', 'orange_light', 'stop_sign_wait', 'stop_sign', 'yield_pedestrian'):
            last_move = time.time()       # waiting because a traffic rule says so is not being stuck
        if time.time() - last_move > max_stuck:
            max_stuck = time.time() - last_move
            stuck_at = (time.time() - t0, g.x, g.y, st['shield'][-1] if st['shield'] else None, dict(st['planner']))
        dy = math.atan2(math.sin(st['yaw'] - last_yaw), math.cos(st['yaw'] - last_yaw))
        last_yaw = st['yaw']
        net = math.atan2(math.sin(st['yaw'] - turn_start), math.cos(st['yaw'] - turn_start))
        if abs(net) > math.radians(60):          # a turn = 60 deg net heading change
            turns += 1
            print(f'  turn {turns}: {"left" if net > 0 else "right"} at ({g.x:.0f},{g.y:.0f})')
            turn_start = st['yaw']
        if a.goal and goal_time is None and math.hypot(g.x - a.goal[0], g.y - a.goal[1]) < a.goal_tolerance:
            goal_time = time.time() - t0
            print(f'  GOAL reached at t={goal_time:.0f}s')
            break
        if time.time() - tp > 10:
            tp = time.time()
            pl = st['planner']
            print(f'  t={time.time() - t0:3.0f}s  {dist:6.1f} m  planner: v {pl.get("speed")} steer {pl.get("steer")} '
                  f'free {pl.get("free")} right_gap {pl.get("right_clear")}')

    clear = min(st['clear']) if st['clear'] else math.inf
    sh = [r for r in st['shield'] if r]
    interventions = sum(r not in ('clear',) for r in sh) / max(len(sh), 1)
    print(f"top speed {st['vmax']:.1f} m/s ({st['vmax'] * 3.6:.0f} km/h)")
    print(f'\ndistance {dist:.0f} m, turns {turns}, min clearance car body -> obstacle {clear:.2f} m, '
          f'longest stop {max_stuck:.1f} s, shield limited speed {100 * interventions:.0f}% of the time')
    lane = st['lane']
    offroad = 100 * float(np.mean([f > 0.5 for f, _ in lane])) if lane else float('nan')
    centre = 100 * float(np.mean([c for _, c in lane])) if lane else float('nan')
    print(f'lane discipline ({len(lane)} frames): car off the road {offroad:.1f}% of the time, '
          f'on the centre line {centre:.1f}% of the time')
    if planner:
        planner.terminate()
    checks = [] if not a.goal else [('goal reached', goal_time is not None)]
    if a.goal:
        a.min_distance, a.min_turns = 0, 0
    checks += [('distance', dist >= a.min_distance), ('turns', turns >= a.min_turns),
              ('stays on road', lane and offroad < 2.0), ('own side of the road', lane and centre < 2.0),
              ('clearance', clear >= a.min_clearance), ('not stuck', max_stuck <= a.max_stuck)]
    if 'min_clear' in st:
        c, tc, x, y, pl, pb, pw = st['min_clear']
        print(f'  closest approach {c:.2f} m at t={tc - t0:.0f}s ({x:.1f},{y:.1f}) planner {pl}')
        print(f'  closest obstacle point: car frame (x, y, height) {pb}, world {pw}')
    if stuck_at:
        print(f'  longest stop ended/at t={stuck_at[0]:.0f}s ({stuck_at[1]:.1f},{stuck_at[2]:.1f}) '
              f'shield {stuck_at[3]} planner {stuck_at[4]}')
    for name, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    ok = all(ok for _, ok in checks)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
