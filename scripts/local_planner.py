#!/usr/bin/env python3
"""Local planner node: lidar obstacles -> steering + speed -> /cmd_drive_raw (-> safety shield).

Input : /perception/obstacle_points  (base_link, everything that is not road)
        /perception/road_boundary    (base_link, VIRTUAL walls from the camera: off-road ground and
                                     the yellow centre line, so turns stay on our side of the road)
        /odometry/filtered           (EKF, for the goal transform)
        /route/path                  (optional lane-level route from route_planner, odom frame)
        /goal_pose                   (fallback when no lane map exists: steer toward the goal)
Output: /cmd_drive_raw               AckermannDriveStamped  -> scripts/safety_shield.py -> /cmd_drive
        /planner/arcs                MarkerArray: candidate arcs (grey), chosen arc (green), goal
        /planner/status              String JSON: chosen steer/speed, free distance, reason

The logic is auto_car_core.planner.plan (unit-tested in test/test_planner_logic.py).
"""
import json
import math

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from auto_car_core.frenet import FrenetParams, plan_frenet
from auto_car_core.livemap import LiveMap
from auto_car_core.safety import path_static_distance, static_distance
from auto_car_core.planner import PlannerParams, passability, plan
from auto_car_core.safety import Params, path_poses


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class LocalPlanner(Node):
    def __init__(self, name='local_planner'):
        super().__init__(name)
        self.pp = PlannerParams(**{k: self.declare_parameter(k, v).value for k, v in PlannerParams().__dict__.items()})
        self.ps = Params()
        self.goal_tolerance = self.declare_parameter('goal_tolerance', 4.0).value
        # Drive only toward a destination (route). Free road-following without one is off by default:
        # an autonomous car without a destination parks; it does not wander.
        self.free_drive = self.declare_parameter('free_drive', False).value
        # 'frenet' (default): Frenet-frame trajectory planner along the lane route (Werling et al.);
        # 'arcs': the older arc-sampling planner
        self.planner_type = self.declare_parameter('planner_type', 'arcs').value
        self.cruise_speed = self.declare_parameter('cruise_speed', 11.1).value   # m/s (40 km/h) open road; curves, traffic, tight spaces slow it
        self.parking = None                     # active parking manoeuvre (from /route/parking)
        self.create_subscription(String, '/route/parking', self.on_parking, 5)
        self.pub_park_next = self.create_publisher(String, '/route/parking_next', 5)
        self.fp = FrenetParams()
        self.arrived = False
        self.points = None
        self.pose = None            # (x, y, yaw) in odom
        self.goal = None            # (x, y) in odom
        self.prev_steer = 0.0
        self.v = 0.0
        self.road_width, self.road_width_t = None, -1e9   # last measured width (for junctions)
        self.pub = self.create_publisher(AckermannDriveStamped, '/cmd_drive_raw', 10)
        self.pub_arcs = self.create_publisher(MarkerArray, '/planner/arcs', 5)
        self.pub_plan_path = self.create_publisher(Path, '/planner/path', 5)   # checked by the safety shield
        self.pub_status = self.create_publisher(String, '/planner/status', 10)
        self.road = np.zeros((0, 3), np.float32)
        self.create_subscription(PointCloud2, '/perception/obstacle_points', self.on_points, 5)
        self.create_subscription(PointCloud2, '/perception/road_boundary', self.on_road, 5)
        self.kerb = np.zeros((0, 3), np.float32)
        self.create_subscription(PointCloud2, '/perception/kerb_points', lambda m: setattr(
            self, 'kerb', point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))), 5)
        self.rule_walls = np.zeros((0, 3), np.float32)        # stop lines from traffic_rules
        self.speed_limit = None
        self.create_subscription(PointCloud2, '/behaviour/stop_points', lambda m: setattr(
            self, 'rule_walls', point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))), 5)
        self.create_subscription(Float32, '/behaviour/speed_limit', lambda m: setattr(self, 'speed_limit', m.data), 5)
        self.road_type = None              # 'highway' allows highway speed (route_planner /route/road)
        self.create_subscription(String, '/route/road', lambda m: setattr(
            self, 'road_type', json.loads(m.data).get('type')), 5)
        self.driver_limit = None           # driver monitor: slow down / stop when the driver is unresponsive
        from auto_car_core.speed_rl import SpeedController
        self.speed_ctrl = SpeedController()  # RL speed control (always on; old rule if no model)
        self.get_logger().info(f'speed control: {self.speed_ctrl.source}')
        self.create_subscription(Float32, '/driver/speed_limit', lambda m: setattr(self, 'driver_limit', m.data), 5)
        self.driver_state = 'OK'
        self.create_subscription(String, '/driver/status', lambda m: setattr(
            self, 'driver_state', json.loads(m.data).get('state', 'OK')), 5)
        self.stop_distance = 1e9
        self.create_subscription(Float32, '/behaviour/stop_distance', lambda m: setattr(self, 'stop_distance', m.data), 5)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 20)
        # RL sudden-obstacle handling + RL steering residual (always on; inactive without models)
        from auto_car.msg import TrackArray
        from auto_car_core.evade_rl import Evader
        from auto_car_core.steer_rl import SteerResidual
        self.tracks = []
        self.create_subscription(TrackArray, '/perception/tracks', lambda m: setattr(self, 'tracks', [
            (tr.position.x, tr.position.y, tr.velocity.x, tr.velocity.y, tr.label) for tr in m.tracks]), 5)
        self.evader = Evader()
        # LIVE free-space sensing: count lidar points inside every bay within 12 m
        self.bays_odom, self.bay_state = [], {}
        self.create_subscription(String, '/map/parking_bays', lambda m: setattr(
            self, 'bays_odom', json.loads(m.data)), QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                                          reliability=ReliabilityPolicy.RELIABLE))
        self.pub_occ = self.create_publisher(String, '/parking/occupancy', 10)
        self.create_timer(0.5, self.scan_bays)
        from auto_car_core.park_rl import Parker
        self.parker = Parker()
        self.get_logger().info(f'parking: {self.parker.source}')
        self.steer_rl = SteerResidual()
        self.evade = (False, None, 0.0, 'off')
        self.get_logger().info(f'sudden obstacles: {self.evader.source}, steering: {self.steer_rl.source}')
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 5)
        self.route = None           # Nx2 odom
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Path, '/route/path', self.on_route, latched)
        self.lane_walls = np.zeros((0, 3), np.float32)       # odom: left edge of our lane (map)
        self.create_subscription(PointCloud2, '/route/lane_walls', lambda m: setattr(
            self, 'lane_walls', point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))), latched)
        self.create_timer(0.1, self.loop)
        self.get_logger().info(f'local_planner: {self.pp.n_arcs} arcs, v_max {self.pp.v_max} m/s, '
                               f'keep {self.pp.keep_right} m from the right edge')

    def on_points(self, m):
        # lidar builds the LIVE MAP of the surroundings; the planner plans against the map
        # (obstacles confirmed over several scans), never against a single raw scan
        raw = point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))
        self.points_raw = raw                                # what the safety shield checks
        if self.pose is None:
            self.points = raw
            return
        if not hasattr(self, 'livemap'):
            self.livemap = LiveMap()
        self.livemap.update(raw, self.pose)
        self.points = self.livemap.occupied(self.pose)

    def on_road(self, m):
        self.road = point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))

    def on_route(self, m):
        had_route = self.route is not None
        self.route = (np.array([[p.pose.position.x, p.pose.position.y] for p in m.poses])
                      if len(m.poses) >= 2 else None)
        if self.route is not None:
            self.goal = None        # the route encodes the goal
            self.arrived = False
            self.parking = None     # a new destination cancels a finished/ongoing parking
        elif had_route:
            # route cleared = destination reached: STOP and stay parked until a new destination.
            # (Before, the planner fell back to free road-following, drove on past a road-end sign
            # and into the sand.)
            self.arrived = True
            self.get_logger().info('destination reached: PARKED at the kerb side, waiting for a new destination')

    def route_in_base(self):
        if self.route is None or self.pose is None:
            return None
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = self.route[:, 0] - x, self.route[:, 1] - y
        return np.c_[c * dx + s * dy, -s * dx + c * dy]

    def lane_walls_in_base(self, reach=40.0):
        if len(self.lane_walls) == 0 or self.pose is None:
            return np.zeros((0, 3), np.float32)
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        dx, dy = self.lane_walls[:, 0] - x, self.lane_walls[:, 1] - y
        bx, by = c * dx + s * dy, -s * dx + c * dy
        keep = (np.hypot(bx, by) < reach) & (bx > -6.0)
        near = np.abs(bx) < 8.0
        if near.any() and np.median(by[near]) < 0.0:
            # The lane line (left edge of OUR lane) is on the car's RIGHT: the car is already on the
            # wrong side (oncoming lane). The line must not block the way back - drop it, so the
            # route/lane-exit terms steer the car straight back into its own lane.
            if not getattr(self, '_wrong_side_logged', False):
                self.get_logger().warn('car is left of its lane line (oncoming lane): steering back')
                self._wrong_side_logged = True
            return np.zeros((0, 3), np.float32)
        self._wrong_side_logged = False
        return np.c_[bx[keep], by[keep], np.full(keep.sum(), 0.5)].astype(np.float32)

    def road_on_route(self, corridor=1.3):
        """Camera road-boundary (virtual wall) points, minus those inside our lane along the route.
        The lane map knows the road continues: in a tunnel the lamps paint bright patches across
        the asphalt, the camera called them off-road and walled the lane off at the tunnel entrance.
        Walls outside the corridor (real road edge, opposite lane) still keep the car in place.
        The corridor is 1.3 m: INSIDE the lane only. (2.5 m also removed the real road edge, ~1.75 m
        from the lane centre: the car then cut corners over the kerb.)"""
        route = self.route_in_base()
        boundary = np.r_[self.road, self.kerb] if len(self.kerb) else self.road
        if route is None or len(route) < 2 or len(boundary) == 0:
            return boundary
        seg = np.diff(route, axis=0)
        n = np.maximum(np.ceil(np.hypot(*seg.T) / 0.5).astype(int), 1)
        dense = np.vstack([route[i] + seg[i] * np.linspace(0, 1, k, endpoint=False)[:, None]
                           for i, k in enumerate(n)] + [route[-1:]])
        from scipy.spatial import cKDTree
        d, _ = cKDTree(dense).query(boundary[:, :2])
        return boundary[d > corridor]

    def on_odom(self, m):
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
        self.v = m.twist.twist.linear.x

    def on_goal(self, m):
        # map and odom coincide at start-up (SLAM origin = EKF origin); good enough for a goal hint
        self.goal = (m.pose.position.x, m.pose.position.y)
        self.get_logger().info(f'new goal ({self.goal[0]:.1f}, {self.goal[1]:.1f}) in {m.header.frame_id}')

    def goal_in_base(self):
        if self.goal is None or self.pose is None:
            return None
        x, y, yaw = self.pose
        dx, dy = self.goal[0] - x, self.goal[1] - y
        if math.hypot(dx, dy) < self.goal_tolerance:
            self.get_logger().info('goal reached')
            self.goal = None
            return 'reached'
        c, s = math.cos(yaw), math.sin(yaw)
        return (c * dx + s * dy, -s * dx + c * dy)

    def reverse_recovery(self, steer, v, info, now):
        """Back up when WEDGED: the route wants the car to turn but its corner is against a wall /
        kerb post (planned path blocked within 0.8 m), it has stood still for 3 s and no traffic
        rule holds it. Like a driver: reverse ~2 m with the steering turned the other way (the nose
        swings away from the obstacle), then drive on. The shield checks the space behind."""
        rv = getattr(self, 'rev', None)
        if rv is not None:                                     # manoeuvre in progress
            x, y, _ = self.pose
            moved = math.hypot(x - rv['x0'], y - rv['y0'])
            fwd = rv.get('mode') == 'creep'
            done = moved >= (1.0 if fwd else 2.0) or now - rv['t0'] > 6.0 or \
                (now - rv['t0'] > 2.5 and (self.v < 0.05 if fwd else self.v > -0.05))
            if done:
                self.rev = None
                self.wedged_since = None
                if not fwd and moved < 0.3:
                    self.rear_blocked_t = now                  # something right behind us (traffic)
                self.get_logger().info(f"{'creep forward' if fwd else 'reverse'} manoeuvre done ({moved:.1f} m)")
            else:
                return rv['steer'], (0.4 if fwd else -0.8), dict(info, reason='creeping' if fwd else 'reversing')
            return steer, v, info
        best = info.get('best') or {}
        wedged = False
        if (info.get('reason') in ('ok_lane_follow', 'lane_blocked') and 'x' in best and abs(self.v) < 0.1
                and v < 0.05 and self.stop_distance > 1e8 and getattr(self, 'points_raw', None) is not None):
            d = path_static_distance(self.points_raw, np.c_[best['x'], best['y']], self.ps, max_travel=5.0)
            wedged = d is not None and d < 0.8
        if not wedged:
            self.wedged_since = None
            return steer, v, info
        self.wedged_since = getattr(self, 'wedged_since', None) or now
        if now - self.wedged_since > 3.0 and self.pose is not None:
            rear_blocked = now - getattr(self, 'rear_blocked_t', -1e9) < 20.0
            if rear_blocked:
                # cannot back up (traffic queued behind): creep forward along the steering that
                # leaves the most room, to free the corner; else wait for the traffic behind
                best_st, best_d = None, 0.6
                for cand in np.arange(-0.45, 0.4501, 0.05):
                    d = static_distance(self.points_raw, float(cand), self.ps, max_travel=3.0, step=0.1)
                    if d > best_d + 0.05 or (best_st is not None and abs(d - best_d) <= 0.05 and abs(cand - steer) < abs(best_st - steer)):
                        best_st, best_d = float(cand), d
                if best_st is not None:
                    self.rev = {'x0': self.pose[0], 'y0': self.pose[1], 't0': now, 'steer': best_st, 'mode': 'creep'}
                    self.get_logger().warn(f'wedged and blocked behind: creeping forward, steer {best_st:+.2f} ({best_d:.1f} m free)')
                    return best_st, 0.4, dict(info, reason='creeping')
                if now - self.wedged_since < 8.0:
                    return steer, 0.0, dict(info, reason='waiting_for_traffic_behind')
                self.rear_blocked_t = -1e9                     # waited: try reversing again
            rsteer = float(-np.sign(steer) * 0.4) if abs(steer) > 0.05 else 0.0
            self.rev = {'x0': self.pose[0], 'y0': self.pose[1], 't0': now, 'steer': rsteer}
            self.get_logger().warn(f'wedged (path blocked within 0.8 m for 3 s): reversing 2 m, steer {rsteer:+.2f}')
            return rsteer, -0.8, dict(info, reason='reversing')
        return steer, v, info

    def scan_bays(self):
        """Free-space check of the parking bays the car passes (as parking-assist sensors do):
        lidar points 0.2-1.9 m above the road inside a bay's box (shrunk 0.3 m) -> TAKEN (>= 15
        points, two scans in a row); none while the bay is within 12 m -> FREE."""
        q = getattr(self, 'points_raw', None)
        if not self.bays_odom or self.pose is None or q is None or len(q) == 0:
            return
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        q = q[(q[:, 2] > -0.12) & (q[:, 2] < 1.6)]
        wx, wy = x + c * q[:, 0] - s * q[:, 1], y + s * q[:, 0] + c * q[:, 1]
        for b in self.bays_odom:
            d = math.hypot(b['x'] - x, b['y'] - y)
            if d > 12.0:
                continue
            cb, sb = math.cos(b['yaw']), math.sin(b['yaw'])
            bx = (wx - b['x']) * cb + (wy - b['y']) * sb
            by = -(wx - b['x']) * sb + (wy - b['y']) * cb
            n = int(((np.abs(bx) < b['length'] / 2 - 0.3) & (np.abs(by) < b['width'] / 2 - 0.3)).sum())
            key = (b['lot'], b['index'])
            occ = n >= 15
            hist = self.bay_state.setdefault(key, [])
            hist.append(occ)
            del hist[:-2]
            if len(hist) == 2 and hist[0] == hist[1]:
                self.pub_occ.publish(String(data=json.dumps({'lot': key[0], 'index': key[1], 'occupied': occ,
                                                             'points': n})))

    def on_parking(self, m):
        d = json.loads(m.data)
        d.update(phase='align', t0=None, t_phase=None)
        self.parking = d
        self.parker.start()
        self.get_logger().info(f"parking manoeuvre started: {d['mode']} into bay {d['lot']}/{d['index']}")

    def parking_step(self, now):
        """One control step of the parking manoeuvre -> (steer, speed, reason) or None when done.
        Bay frame: target base_link = bay centre - 1.45 m along the bay heading (car centred).
        lon/lat = car base_link relative to the target (lat > 0: car left of the bay)."""
        P = self.parking
        if P is None or self.pose is None:
            return None
        x, y, yaw = self.pose
        yb = P['yaw']
        t = np.array([math.cos(yb), math.sin(yb)])
        n = np.array([-t[1], t[0]])
        tgt = np.array([P['x'], P['y']]) - 1.45 * t
        rel = np.array([x, y]) - tgt
        lon, lat = float(rel @ t), float(rel @ n)
        dyaw = math.atan2(math.sin(yaw - yb), math.cos(yaw - yb))
        R = self.ps.wheelbase / math.tan(0.45)
        P['t0'] = P['t0'] or now
        P['t_phase'] = P['t_phase'] or now
        if now - P['t0'] > 90.0 and P['phase'] != 'parked':
            self.get_logger().warn('parking: took too long, stopping where the car is')
            P['phase'] = 'parked'
        if P['phase'] == 'align' and not P.get('checked') and getattr(self, 'points_raw', None) is not None:
            # is the bay really free? lidar points inside the bay box (0.2 .. 2.2 m high)
            P['checked'] = True
            q = self.points_raw
            c_, s_ = math.cos(yaw), math.sin(yaw)
            wx, wy = x + c_ * q[:, 0] - s_ * q[:, 1], y + s_ * q[:, 0] + c_ * q[:, 1]
            bx_, by_ = (wx - P['x']) * t[0] + (wy - P['y']) * t[1], -(wx - P['x']) * t[1] + (wy - P['y']) * t[0]
            inside = (np.abs(bx_) < 2.6) & (np.abs(by_) < 1.3) & (q[:, 2] > -0.12) & (q[:, 2] < 1.9)
            if int(inside.sum()) > 15:
                self.get_logger().warn(f"parking: bay {P['lot']}/{P['index']} is OCCUPIED ({int(inside.sum())} lidar points): next free bay")
                self.pub_park_next.publish(String(data=json.dumps({'lot': P['lot'], 'index': P['index']})))
                self.parking = None
                return 0.0, 0.0, 'parking_bay_occupied'
        hold = float(np.clip(-1.5 * dyaw - 0.0, -0.3, 0.3))          # keep the heading straight
        ph = P['phase']
        # RL PARKING (goal-conditioned SAC + HER, rl/train_park.py): drives the whole parallel
        # manoeuvre - forward / reverse corrections like a driver - until it is in the bay
        if ph != 'parked' and P.get('mode') == 'parallel' and self.parker.model is not None:
            st, v, done = self.parker.step(self.pose, (P['x'], P['y'], yb), self.v, self.prev_steer,
                                           getattr(self, 'points_raw', None))
            if done:
                P['phase'], P['t_phase'] = 'parked', now
                return 0.0, 0.0, 'parked'
            return st, v, 'parking_rl'

        def go(nxt):
            P['phase'], P['t_phase'] = nxt, now
            self.get_logger().info(f'parking: {nxt} (lon {lon:+.2f} lat {lat:+.2f} yaw {math.degrees(dyaw):+.1f} deg)')

        if ph == 'align':
            if P['mode'] == 'parallel' and abs(lat) > 1.0:
                th = math.acos(max(-1.0, 1.0 - abs(lat) / (2 * R)))
                want = 2 * R * math.sin(th)
            else:
                want = 0.0
            e = want - lon                                            # + = must go forward
            if abs(e) < 0.2 and abs(self.v) < 0.1:
                P['theta'] = math.acos(max(-1.0, 1.0 - abs(lat) / (2 * R))) if want > 0 else 0.0
                P['side'] = 1.0 if lat > 0 else -1.0
                go('arc1' if want > 0 else 'centre')
                return hold, 0.0, 'parking_align'
            v = float(np.clip(0.6 * e, -0.6, 0.6))
            v = v if abs(v) > 0.25 else math.copysign(0.25, e)
            return (hold if v > 0 else -hold), v, 'parking_align'
        if ph == 'arc1':
            # reverse at full lock toward the bay until the car has turned theta
            st = -P['side'] * 0.45
            if abs(dyaw) >= P['theta'] - 0.01:
                go('arc2')
                return -st, 0.0, 'parking_arc'
            return st, -0.5, 'parking_arc'
        if ph == 'arc2':
            st = P['side'] * 0.45
            if abs(dyaw) < 0.03 or (now - P['t_phase'] > 3.0 and dyaw * P['side'] * -1 < 0):
                go('centre')
                return 0.0, 0.0, 'parking_arc'
            return st, -0.5, 'parking_arc'
        if ph == 'centre':
            if abs(lon) < 0.25 and abs(self.v) < 0.1:
                go('parked')
                return 0.0, 0.0, 'parked'
            v = float(np.clip(-0.6 * lon, -0.5, 0.5))
            v = v if abs(v) > 0.2 else math.copysign(0.2, -lon)
            return (hold if v > 0 else -hold), v, 'parking_centre'
        if ph == 'parked':
            if not P.get('logged'):
                P['logged'] = True
                self.get_logger().info(f'PARKED in bay {P["lot"]}/{P["index"]}: lon {lon:+.2f} m, lat {lat:+.2f} m, '
                                       f'heading {math.degrees(dyaw):+.1f} deg')
            return 0.0, 0.0, 'parked'
        return None

    def park_path(self, route):
        """PARKING at the destination: over the last 25 m the path eases 0.5 m to the right, toward
        the kerb (the car's side stays ~0.25 m off the road edge), and the speed falls so that the car
        stops at the end of the route. Returns (path, remaining metres)."""
        if route is None or len(route) < 2:
            return route, None
        i0 = int(np.argmin(np.hypot(route[:, 0], route[:, 1])))
        seg = np.hypot(*np.diff(route, axis=0).T)
        to_end = np.r_[np.cumsum(seg[::-1])[::-1], 0.0]           # metres from each point to the end
        remaining = float(to_end[i0])
        if remaining > 25.0:
            return route, remaining
        t = np.gradient(route, axis=0)
        t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
        shift = 0.5 * np.clip((25.0 - to_end) / 15.0, 0.0, 1.0)    # 0 at 25 m, full 0.5 m from 10 m
        return route + shift[:, None] * np.c_[t[:, 1], -t[:, 0]], remaining

    def decide(self, pts, goal, now):
        """(steer, speed, info) from the merged obstacle points. rl_planner.py overrides this."""
        # ONE steady cruise speed for every task (normal and emergency drives): 40 km/h. Only turns,
        # traffic rules and obstacles make the car slower.
        self.pp.v_max = float(self.get_parameter("cruise_speed").value)   # live: ros2 param set
        route, self.remaining = self.park_path(self.route_in_base())
        if route is not None and len(route) >= 3 and self.parking is None:
            # RL sudden-obstacle: lateral target relative to the lane centre -> shift the route
            near = route[max(int(np.argmin(np.hypot(route[:, 0], route[:, 1]))) - 2, 0):][:80]
            self.evade = self.evader.step(near, self.points, self.tracks, self.v,
                                          getattr(self, 'a_cmd_prev', 0.0))
            if self.evade[0] and abs(self.evade[2]) > 0.05:
                g = np.gradient(route, axis=0)
                g /= np.maximum(np.hypot(*g.T), 1e-9)[:, None]
                nrm = np.c_[-g[:, 1], g[:, 0]]
                ramp = np.clip(route[:, 0] / 6.0, 0.0, 1.0)[:, None]      # ease in over 6 m
                route = route + self.evade[2] * ramp * nrm
        if self.planner_type == 'frenet' and route is not None and len(route) >= 2:
            road = self.road_on_route() if (len(self.road) or len(self.kerb)) else None
            return plan_frenet(route, self.points, road, ego_v=self.v, highway=self.road_type == 'highway',
                               fp=self.fp, ps=self.ps)
        hint = self.road_width if now - self.road_width_t < 5.0 else None
        # avoidance (arcs off the route line) only after the lane has been blocked for 4 s with the
        # car stopped: a parked car / obstacle that really stays, not a lidar ripple
        avoid = (getattr(self, 'blocked_since', None) is not None and now - self.blocked_since > 4.0)
        st, v, info = plan(pts, self.prev_steer, goal, self.pp, self.ps, road_width=hint,
                           path=route, n_physical=len(self.points), highway=self.road_type == 'highway',
                           ego_v=self.v, avoid=avoid, raw_points=getattr(self, 'points_raw', None),
                           nudge_prev=getattr(self, 'nudge_prev', 0.0),
                           road_edges=np.r_[self.road, self.kerb] if (len(self.road) or len(self.kerb)) else None)
        self.nudge_prev = float(info.get('nudge', 0.0) or 0.0)
        if info.get('reason') == 'lane_blocked' and abs(self.v) < 0.2:
            self.blocked_since = getattr(self, 'blocked_since', None) or now
        elif info.get('reason') == 'ok_lane_follow':
            self.blocked_since = None
        return st, v, info

    def loop(self):
        if self.points is None:
            return
        now = self.get_clock().now().nanoseconds * 1e-9
        goal = self.goal_in_base()
        if self.route is None and (getattr(self, 'arrived', False) or not self.free_drive):
            # no destination: hold still (free road-following only if explicitly enabled)
            steer, v, info = self.prev_steer, 0.0, {'reason': 'arrived' if getattr(self, 'arrived', False)
                                                    else 'no_destination', 'arcs': [], 'best': None}
        elif goal == 'reached':
            steer, v, info = self.prev_steer, 0.0, {'reason': 'goal_reached', 'arcs': [], 'best': None}
        else:
            pts_nowall = np.r_[self.points, self.road_on_route()] if len(self.road) else self.points
            walls = self.lane_walls_in_base()
            if self.evade[0] and self.evade[2] > 0.75:
                walls = np.zeros((0, 3))          # RL evasion over the centre line (envelope: no oncoming)
            pts = np.r_[pts_nowall, walls] if len(walls) else pts_nowall   # map lane edge: no oncoming lane
            # The stop-line wall (red light / stop sign / pedestrian) is NOT given to the arc planner:
            # every straight arc hits it early, so the arcs that turn sideways along it looked best
            # and the car swung across the centre line while stopping. The stop is enforced by
            # stop_distance below (and by the shield, which keeps the wall); the planner keeps
            # following the lane, so the car stops straight in its lane.
            now = self.get_clock().now().nanoseconds * 1e-9
            steer, v, info = self.decide(pts, goal, now)
            if info.get('reason', '').startswith('no_free_arc') or info.get('reason', '').endswith('_virtual_ignored'):
                if len(walls):
                    # first drop only the map lane lines, KEEP the camera road edge (it keeps the car
                    # on the road); the planner's own fallback ignores all virtual walls only after that
                    s2, v2, i2 = self.decide(pts_nowall, goal, now)
                    if not i2.get('reason', '').startswith('no_free_arc'):
                        steer, v, info = s2, v2, i2
                        info['reason'] = info['reason'] + '_lane_lines_ignored'
            if info.get('road_width'):
                w = info['road_width']
                self.road_width = w if self.road_width is None else 0.8 * self.road_width + 0.2 * w
                self.road_width_t = now
        # RL steering residual (<= 0.05 rad) on top of the planner while following the route
        if info.get('reason') == 'ok_lane_follow' and self.route is not None and self.parking is None:
            rb = self.route_in_base()
            if rb is not None and len(rb) >= 3:
                i0 = int(np.argmin(np.hypot(rb[:, 0], rb[:, 1])))
                steer = steer + self.steer_rl.residual(rb[max(i0 - 2, 0):i0 + 40], self.v, self.prev_steer, steer)
        # smooth steering like a driver: at most 0.08 rad change per 0.1 s cycle (no jerks / waves)
        steer = float(np.clip(steer, self.prev_steer - 0.08, self.prev_steer + 0.08))
        self.prev_steer = steer
        # turning: the ONLY reason to drive slower than the steady cruise speed is the turn itself.
        # Lateral acceleration v^2 * tan(steer) / wheelbase stays <= 1.5 m/s^2 (comfortable, like a driver)
        # smoothed steering (0.5 s): a momentary steering wiggle must not dip the speed
        self.steer_avg = 0.8 * getattr(self, 'steer_avg', 0.0) + 0.2 * abs(steer)
        k_now = math.tan(self.steer_avg) / self.ps.wheelbase
        if k_now > 1e-3:
            v = min(v, max(2.5, math.sqrt(1.5 / k_now)))
        # comfort: the commanded speed rises at most 1.2 m/s per second (braking is never limited)
        # ---------------- SPEED: target from every limit, then a GRADUAL change towards it ----------
        self.dbg = {'v_plan': round(float(v), 2)}
        v_target = v
        if self.speed_limit is not None:
            v_target = min(v_target, float(self.speed_limit))
        if self.driver_limit is not None:
            v_target = min(v_target, float(self.driver_limit))
        rule_stop = self.stop_distance < 1e8                   # red light / stop sign / give way
        if rule_stop:
            # stop 1 m before the line, braking comfortably (1.5 m/s^2) and early
            room = self.stop_distance - 1.0
            v_target = min(v_target, 0.0 if room <= 0 else math.sqrt(2 * 1.5 * room))
        v_prev = getattr(self, 'v_cmd_prev', 0.0)
        a_prev = getattr(self, 'a_cmd_prev', 0.0)
        dt = 0.1
        # comfortable limits: +1.0 m/s^2 speeding up, -1.5 m/s^2 slowing down (curves, traffic,
        # limits); stops for rules / something in the lane may use -3.0 m/s^2. Jerk (change of
        # acceleration) <= 1 m/s^3 (2 m/s^3 when braking for a stop) so speed eases in and out.
        # Emergency braking is the safety shield's job and is never smoothed.
        firm = rule_stop or info.get('reason') not in ('ok_lane_follow',)
        # RL SPEED CONTROL (auto_car_core/speed_rl.py, SAC from rl/train_speed.py): HOW to reach the
        # target speed. Never above the target (hard limit); old rule if the model is missing.
        # free distance in the lane: the planner measures it from the rear axle along the route and
        # caps it at its horizon (45 m, 120 m on the highway) -> front bumper, 'nothing' = 200 m
        lf = info.get('free')
        free_obj = 200.0 if (lf is None or not math.isfinite(lf) or 44.5 <= lf < 45.5 or lf >= 119.5) else max(float(lf) - 4.05, 0.0)
        a_des = self.speed_ctrl.accel(v_prev, a_prev, v_target, free_obj,
                                      float(self.stop_distance) if rule_stop else 1e9,
                                      float(self.speed_limit) if self.speed_limit else 13.9, dt=dt, firm=firm)
        self.dbg.update(v_target=round(float(v_target), 2), a_speed=round(float(a_des), 2))
        if self.evade[0] and self.evade[1] is not None:
            a_des = min(a_des, float(self.evade[1]))        # RL evasion may brake harder, never faster
        self.dbg.update(evade=[bool(self.evade[0]), None if self.evade[1] is None else round(float(self.evade[1]), 2),
                               round(float(self.evade[2]), 2), self.evade[3]])
        jerk = 2.0 if (firm and a_des < a_prev) else 1.0
        a_cmd = a_prev + float(np.clip(a_des - a_prev, -jerk * dt, jerk * dt))
        v = v_prev + a_cmd * dt
        if (v - v_target) * (v_prev - v_target) <= 0 and abs(v - v_target) < 0.3:
            v, a_cmd = v_target, 0.0                           # reached the target: settle
        if v_target <= 0.05 and v_prev <= 0.3:
            v, a_cmd = 0.0, 0.0                                # hold at standstill
        v = max(0.0, v)
        self.a_cmd_prev = a_cmd
        if self.parking is not None and self.route is None:
            out = self.parking_step(now)
            if out is not None:
                steer, v, reason = out
                info = dict(info, reason=reason)
                self.prev_steer = steer
                self.v_cmd_prev = max(v, 0.0)
                cmd = AckermannDriveStamped()
                cmd.header.stamp = self.get_clock().now().to_msg()
                cmd.header.frame_id = 'base_link'
                cmd.drive.speed, cmd.drive.steering_angle = float(v), float(steer)
                self.pub.publish(cmd)
                self.pub_status.publish(String(data=json.dumps({'reason': reason, 'steer': round(steer, 3),
                                                                'speed': round(v, 2), 'parking': self.parking['phase']})))
                return
        rem = getattr(self, 'remaining', None)
        if rem is not None and rem < 25.0 and self.route is not None:
            # parking: slow down smoothly to stop at the end of the route (kerb side)
            v = min(v, max(0.0, 0.45 * (rem - 1.0)), 4.0)
            if rem < 1.5:
                v = 0.0
                if not getattr(self, 'parked', False):
                    self.parked = True
                    self.get_logger().info('PARKED at the destination (pulled over to the kerb side)')
                info = dict(info, reason='parked')
        else:
            self.parked = False
        steer, v, info = self.reverse_recovery(steer, v, info, now)
        self.v_cmd_prev = max(v, 0.0)
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'
        cmd.drive.speed, cmd.drive.steering_angle = float(v), float(steer)
        self.pub.publish(cmd)
        best = info.get('best') or {}
        # Car-size check of the path ahead (lidar points only): ceiling vs car height, objects vs
        # ground clearance, gaps vs car width. Logged whenever the way ahead is not passable.
        pas = passability(self.points, float(steer), self.ps, n_physical=None) if self.points is not None else {}
        if pas.get('verdict', 'pass') != 'pass':
            self.get_logger().info(
                f"way ahead: {pas['verdict']} (ceiling {pas['ceiling']} m vs car {pas['car_height']} m, "
                f"gap {pas['gap']} m vs car {pas['car_width']} m, blocking {pas['blocking']})",
                throttle_duration_sec=2.0)
        self.pub_status.publish(String(data=json.dumps({
            'pass': pas,
            'reason': info['reason'], 'steer': round(steer, 3), 'speed': round(v, 2),
            'road_width': None if self.road_width is None else round(self.road_width, 1),
            'free': round(best.get('free', 0.0), 1), 'right_clear': None if not best or not math.isfinite(
                best.get('right_clear', math.inf)) else round(best['right_clear'], 1),
            'goal': None if not isinstance(goal, tuple) else [round(goal[0], 1), round(goal[1], 1)],
            'nudge': round(float(getattr(self, 'nudge_prev', 0.0)), 2), 'nudge_why': info.get('nudge_why', ''),
            'limit': info.get('limit'), 'dbg': getattr(self, 'dbg', None)})))
        self.publish_arcs(info, goal)
        best = info.get('best') or {}
        if info.get('reason') in ('ok_lane_follow', 'lane_blocked') and 'x' in best and len(best['x']) >= 3:
            pm = Path()
            pm.header.frame_id, pm.header.stamp = 'base_link', self.get_clock().now().to_msg()
            for px, py in zip(best['x'], best['y']):
                ps_ = PoseStamped()
                ps_.pose.position.x, ps_.pose.position.y = float(px), float(py)
                pm.poses.append(ps_)
            self.pub_plan_path.publish(pm)

    def publish_arcs(self, info, goal):
        ma = MarkerArray()
        ma.markers.append(Marker(action=Marker.DELETEALL))
        stamp = self.get_clock().now().to_msg()
        trajs = info.get('trajectories') or []
        best = info.get('best') or {}
        for i, tr in enumerate(trajs + ([best] if 'x' in best else [])):
            chosen = i == len(trajs)
            m = Marker(ns='trajectories', id=i, type=Marker.LINE_STRIP, action=Marker.ADD)
            m.header.frame_id, m.header.stamp = 'base_link', stamp
            m.pose.orientation.w = 1.0
            m.scale.x = 0.3 if chosen else 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = (0.1, 1.0, 0.2, 1.0) if chosen else (0.7, 0.7, 0.7, 0.4)
            for x, y in zip(tr['x'][::2], tr['y'][::2]):
                m.points.append(Point(x=float(x), y=float(y), z=-0.2))
            ma.markers.append(m)
        for i, a in enumerate(info.get('arcs', [])):
            m = Marker(ns='arcs', id=i, type=Marker.LINE_STRIP, action=Marker.ADD)
            m.header.frame_id, m.header.stamp = 'base_link', stamp
            m.pose.orientation.w = 1.0
            chosen = info.get('best') is a
            m.scale.x = 0.25 if chosen else 0.05
            m.color.r, m.color.g, m.color.b, m.color.a = ((0.1, 1.0, 0.2, 1.0) if chosen else
                                                          ((0.6, 0.6, 0.6, 0.5) if a['feasible'] else (0.8, 0.2, 0.2, 0.4)))
            for s in np.linspace(0.0, max(a['free'], 0.5), 20):
                x, y, _ = path_poses(1.0, a['steer'], self.ps, s)
                m.points.append(Point(x=float(x), y=float(y), z=-0.2))
            ma.markers.append(m)
        if isinstance(goal, tuple):
            g = Marker(ns='goal', id=0, type=Marker.SPHERE, action=Marker.ADD)
            g.header.frame_id, g.header.stamp = 'base_link', stamp
            g.pose.position.x, g.pose.position.y, g.pose.orientation.w = goal[0], goal[1], 1.0
            g.scale.x = g.scale.y = g.scale.z = 1.5
            g.color.r, g.color.g, g.color.b, g.color.a = 1.0, 0.8, 0.0, 1.0
            ma.markers.append(g)
        self.pub_arcs.publish(ma)


def main():
    rclpy.init()
    node = LocalPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
