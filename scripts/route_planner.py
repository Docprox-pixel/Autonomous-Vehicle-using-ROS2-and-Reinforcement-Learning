#!/usr/bin/env python3
"""Route planner: goal -> lane-level route on the lane map -> /route/path for the local planner.

Input : /goal_pose (PoseStamped, odom frame; RViz "2D Goal Pose" works)
        /odometry/filtered (EKF), /gps/position + /imu/data (only to align odom with the map)
Output: /route/path     nav_msgs/Path (odom frame) along lane centrelines, junctions included
        /route/markers  MarkerArray (the route in RViz)
        /route/status   String JSON

Map: a SUMO network (worlds/<world>_net/sumo.net.xml), see auto_car_core.lanemap.

Alignment: the map is in world coordinates, the car drives in odom. The EKF takes its heading
from the IMU (absolute), so odom and world only differ by a translation, measured ONCE from the
first GPS fix (the car never uses GPS for driving; this matches "GPS only for the destination").
The route is re-planned if the car ends up more than `replan_distance` from it.
"""
import json
import math
import os

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose, PoseArray, PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from auto_car_core.lanemap import LaneMap


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class RoutePlanner(Node):
    def __init__(self):
        super().__init__('route_planner')
        share = get_package_share_directory('auto_car')
        net = self.declare_parameter('net_file', '').value
        self.replan_distance = self.declare_parameter('replan_distance', 8.0).value
        layout = yaml.safe_load(open(os.path.join(share, 'config', 'sensor_layout.yaml')))
        self.gps_x = next(l for l in layout['lidars'] if l['name'] == 'lidar')['xyz'][0]   # GPS on roof mount
        self.map = None
        if net and os.path.exists(net):
            self.map = LaneMap(net)
            self.get_logger().info(f'lane map: {os.path.basename(os.path.dirname(net))}, '
                                   f'{len(self.map.road_lanes)} lanes, {len(self.map.lanes) - len(self.map.road_lanes)} junction connections')
        else:
            self.get_logger().warn(f'no lane map ({net!r}): routes disabled, the planner drives without one')
        self.places = []
        pois = os.path.join(os.path.dirname(net), 'pois.yaml') if net else ''
        if pois and os.path.exists(pois):
            self.places = yaml.safe_load(open(pois)).get('places', [])
        # parking bays (tools/extract_parking.py): normal trips END IN A BAY, never on the road
        self.bays, self.parking, self.bays_excluded, self.park_dest = [], None, set(), None
        bays_file = os.path.join(os.path.dirname(net), 'parking.yaml') if net else ''
        if bays_file and os.path.exists(bays_file):
            self.bays = yaml.safe_load(open(bays_file)).get('bays', [])
            self.get_logger().info(f'{len(self.places)} places (buildings, shops, hospital) from {os.path.basename(pois)}')
        self.offset = None          # world = odom + offset
        self.pose = None            # odom (x, y, yaw)
        self.gps = self.imu_yaw = None
        self.goal_world = None
        self.path_world = None
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub_path = self.create_publisher(Path, '/route/path', latched)
        self.pub_markers = self.create_publisher(MarkerArray, '/route/markers', latched)
        self.pub_status = self.create_publisher(String, '/route/status', 10)
        self.pub_lines = self.create_publisher(PoseArray, '/route/stop_lines', latched)
        self.pub_map = self.create_publisher(MarkerArray, '/map/lanes', latched)
        # parking bays on the map: green FREE, red TAKEN, blue SELECTED; click one with RViz
        # '2D Goal Pose' to park there
        self.pub_bays = self.create_publisher(MarkerArray, '/map/parking', latched)
        self.pub_bay_data = self.create_publisher(String, '/map/parking_bays', latched)
        self.create_timer(3.0, self.publish_bays)
        # LIVE free-space sensing: the local planner checks every bay it passes with the lidar
        self.bay_live = {}
        self.create_subscription(String, '/parking/occupancy', self.on_occupancy, 10)
        # mapped traffic lights (signals.yaml) in odom, for camera association in traffic_rules
        self.pub_signals = self.create_publisher(PoseArray, '/route/signal_lights', latched)
        self.pub_spat = self.create_publisher(String, '/route/signal_schedule', latched)   # V2I SPaT
        self.pub_junctions = self.create_publisher(String, '/route/junctions', latched)   # right of way
        self.light_info = []
        self.lights = []
        sig = os.path.join(os.path.dirname(net), 'signals.yaml') if net else ''
        if sig and os.path.exists(sig):
            self.light_info = [L for L in (yaml.safe_load(open(sig)) or {}).get('lights', []) if L['type'] == 'GenericTrafficLight']
            self.lights = [(L['x'], L['y'], L.get('yaw', 0.0)) for L in (yaml.safe_load(open(sig)) or {}).get('lights', [])]
        # left edge of our lane along the route (centre line / kerb): virtual walls for the planner
        from sensor_msgs.msg import PointCloud2
        self.pub_walls = self.create_publisher(PointCloud2, '/route/lane_walls', latched)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 20)
        # lane-level map matching: camera distance to the left lane line corrects the lateral
        # alignment (GPS/odometry error of a metre puts the car on the centre line otherwise)
        self.lane_corr = np.zeros(2)          # world-frame correction added to the GPS offset
        self.lane_err = []
        self.cue_marking, self.cue_road, self.cue_kerb, self.lane_cues = None, None, None, {}
        # lane-level matching moves the route only when enabled: off the paved road its edge cues
        # (sand slopes, hills) are wrong and it shifted the route 2.5 m (car drove into the sand).
        # Default: diagnostics only (/route/lane_match); GPS alignment alone positions the route.
        self.lane_match_apply = self.declare_parameter('lane_match_apply', False).value
        self.create_subscription(String, '/perception/lane_marking', self.on_lane_marking, 10)
        self.create_subscription(PointCloud2, '/perception/road_boundary', lambda m: setattr(
            self, 'cue_road', point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))), 5)
        self.create_subscription(PointCloud2, '/perception/kerb_points', lambda m: setattr(
            self, 'cue_kerb', point_cloud2.read_points_numpy(m, field_names=('x', 'y', 'z'))), 5)
        self.pub_cues = self.create_publisher(String, '/route/lane_match', 5)
        self.create_timer(0.2, self.lane_match_tick)
        self.create_subscription(PointStamped, '/gps/position', lambda m: setattr(self, 'gps', m.point), 5)
        self.create_subscription(Imu, '/imu/data', lambda m: setattr(self, 'imu_yaw', yaw_of(m.orientation)), 5)
        self.create_subscription(PoseStamped, '/goal_pose', self.on_goal, 5)
        # Destination by place: kind ('hospital') = nearest of that kind by route length, or a name
        self.create_subscription(String, '/route/go_to', self.on_go_to, 5)
        self.pub_dest = self.create_publisher(String, '/route/destination', latched)
        self.pub_parking = self.create_publisher(String, '/route/parking', 5)
        self.create_subscription(String, '/route/parking_next', self.on_parking_next, 5)
        # which road the car is on: type (one_way / two_way / highway / junction), name, lanes, limit
        self.pub_road = self.create_publisher(String, '/route/road', 5)
        self.pub_label = self.create_publisher(Marker, '/route/car_label', 5)
        self.road_now = None
        self.driver_state = 'OK'
        self.create_subscription(String, '/driver/status', lambda m: setattr(
            self, 'driver_state', json.loads(m.data).get('state', 'OK')), 5)
        self.create_timer(1.0, self.check)

    def on_odom(self, m):
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
        if self.offset is None and self.gps is not None and self.imu_yaw is not None:
            self.offset = self.measured_offset()
            self.get_logger().info(f'odom -> map offset ({self.offset[0]:.2f}, {self.offset[1]:.2f}) from one GPS fix')
            self.publish_map()

    def on_lane_marking(self, m):
        self.cue_marking = json.loads(m.data)

    def road_edges_of(self, lane, P0, t):
        """Map road edges beside `lane` at P0 (unit direction t): offsets (+ left) of the right edge and
        the left edge of the whole carriageway (own-direction lanes + oncoming lanes) from the lane centre."""
        nl = np.array([-t[1], t[0]])
        right, left = -lane.width / 2.0, lane.width / 2.0
        hd = math.atan2(t[1], t[0])
        for ln in self.map.road_lanes:
            if ln is lane:
                continue
            d = np.hypot(*(ln.pts - P0).T)
            k = int(np.argmin(d))
            if d[k] > 14.0:
                continue
            q0, q1 = ln.pts[max(k - 1, 0)], ln.pts[min(k + 1, len(ln.pts) - 1)]
            hl = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
            dh = abs(math.atan2(math.sin(hl - hd), math.cos(hl - hd)))
            if 0.35 < dh < 2.8:
                continue                                       # crossing road, not beside us
            # perpendicular offset of that lane's centre line
            seg = q1 - q0
            off = float((q0 + seg * max(0.0, min(1.0, float((P0 - q0) @ seg) / max(seg @ seg, 1e-9))) - P0) @ nl)
            if abs(off) > 12.0:
                continue
            right = min(right, off - ln.width / 2.0)
            left = max(left, off + ln.width / 2.0)
        return right, left

    @staticmethod
    def side_distance(pts, side):
        """Nearest boundary on one side (side=+1 left, -1 right) from base_link points 0.5-10 m ahead:
        per 1 m slice the closest point on that side, then the median over slices."""
        if pts is None or len(pts) < 5:
            return None, 0
        q = pts[(pts[:, 0] > 0.5) & (pts[:, 0] < 10.0) & (side * pts[:, 1] > 0.6) & (side * pts[:, 1] < 9.0)]
        if len(q) < 5:
            return None, 0
        vals = []
        for xb in np.arange(0.5, 10.0, 1.0):
            sl = q[(q[:, 0] >= xb) & (q[:, 0] < xb + 1.0)]
            if len(sl) >= 2:
                vals.append(float(np.min(side * sl[:, 1])))
        if len(vals) < 4:
            return None, 0
        v = np.array(vals)
        if np.percentile(v, 80) - np.percentile(v, 20) > 1.0:   # not a straight-ish edge (driveway, gap, car)
            return None, len(vals)
        return side * float(np.median(v)), len(vals)

    def lane_match(self):
        """Lane-level map matching WITHOUT relying on the centre line (Bonn 2008 kerb matching,
        TU Braunschweig lane-level map matching): the road edges seen by lidar (kerbs) and camera
        (asphalt boundary) are compared with the edges the map predicts beside our lane; the centre
        line is only one more cue when it is visible. Each cue gives an estimate of the car's TRUE
        offset in its lane; their median (needs agreement) corrects the lateral alignment."""
        if self.path_world is None or self.offset is None or self.pose is None or self.map is None:
            return
        x, y, yaw = self.pose
        bc = np.array([x + 1.45 * math.cos(yaw) + self.offset[0], y + 1.45 * math.sin(yaw) + self.offset[1]])
        hit = self.map.nearest_lane(tuple(bc), yaw, max_angle=math.radians(25), include_internal=False)
        if hit is None or hit[2] > 2.5:
            return                                            # junction / off the map: no lane to match
        lane = hit[0]
        P = self.path_world
        i = int(np.argmin(np.hypot(*(P - bc).T)))
        if i < 6 or i > len(P) - 7 or np.hypot(*(P[i] - bc)) > 3.0:
            return
        a, b = P[i - 6], P[i + 6]
        h1, h2 = math.atan2(*(P[i] - a)[::-1]), math.atan2(*(b - P[i])[::-1])
        if abs(math.atan2(math.sin(h2 - h1), math.cos(h2 - h1))) > math.radians(15):
            return                                            # straight or gentle curve only
        t = P[min(i + 1, len(P) - 1)] - P[max(i - 1, 0)]
        t = t / max(np.hypot(*t), 1e-9)
        hdiff = math.atan2(math.sin(math.atan2(t[1], t[0]) - (yaw)), math.cos(math.atan2(t[1], t[0]) - yaw))
        if abs(hdiff) > math.radians(10):
            return                                            # car not parallel to the lane: base_link y != lateral
        nl = np.array([-t[1], t[0]])
        route_left = float((bc - P[i]) @ nl)                  # where the stack believes the car is (+ left)
        R, L = self.road_edges_of(lane, P[i], t)
        # body centre is 1.45 m ahead of base_link: shift the points so y is measured at the body
        est, used = [], []
        cloud = None
        parts = [c for c in (self.cue_road, self.cue_kerb) if c is not None and len(c)]
        if parts:
            cloud = np.vstack([c[:, :3] for c in parts])
        yr, nr = self.side_distance(cloud, -1)
        yl, nlft = self.side_distance(cloud, +1)
        if yr is not None:
            est.append(R - yr); used.append('right_edge')
        if yl is not None and L - R > lane.width * 1.5:      # left edge only informative with a known width
            est.append(L - yl); used.append('left_edge')
        if yr is not None and yl is not None and abs((yl - yr) - (L - R)) > 1.5:
            est, used = [], []                                # seen road width disagrees with the map: skip
        d = self.cue_marking
        if d and d.get('valid') and abs(d.get('angle', 1.0)) < 0.2 and L - R > lane.width * 1.5:
            est.append(lane.width / 2.0 - d['y_at_body']); used.append('centre_line')
        if not est:
            return
        est = np.array(est)
        if len(est) >= 2 and est.max() - est.min() > 0.8:
            # cues disagree: keep the two closest
            k = int(np.argmin([np.sum(np.abs(est - e)) for e in est]))
            est = est[np.abs(est - est[k]) < 0.8]
        true_left = float(np.median(est))
        e = route_left - true_left
        self.lane_cues = {'used': used, 'true_left': round(true_left, 2), 'believed': round(route_left, 2),
                          'edges_map': (round(R, 2), round(L, 2)),
                          'seen': (None if yr is None else round(yr, 2), None if yl is None else round(yl, 2))}
        if abs(e) < 1.5:
            self.lane_err.append((e * (1.0 if len(est) >= 2 else 0.5), nl))   # single cue: half weight

    def lane_match_tick(self):
        try:
            self.lane_match()
        except Exception as ex:                               # never let perception break routing
            self.get_logger().warn(f'lane_match: {ex}', throttle_duration_sec=10.0)
        if self.lane_cues:
            self.pub_cues.publish(String(data=json.dumps({**self.lane_cues,
                                                          'corr': [round(float(v), 2) for v in self.lane_corr]})))

    def measured_offset(self):
        bx = self.gps.x - self.gps_x * math.cos(self.imu_yaw)
        by = self.gps.y - self.gps_x * math.sin(self.imu_yaw)
        return (bx - self.pose[0], by - self.pose[1])

    def realign(self):
        """Continuous map alignment (the map->odom correction of real stacks): odometry drifts
        (16 m after ~600 m in the village: the route, drawn in odom, then lay 3.5 m left of the real
        lane and the car steered into the oncoming lane). Every second the odom->map offset is
        pulled toward the GPS-measured one (low-pass, GPS noise averaged out); when it has moved by
        > 0.25 m, everything drawn in odom (route, stop lines, lane walls, map) is re-published."""
        if self.offset is None or self.gps is None or self.imu_yaw is None or self.pose is None:
            return
        mx, my = self.measured_offset()
        if not self.lane_match_apply:
            self.lane_err = []                                # diagnostics only: do not accumulate
        if self.lane_err and self.lane_match_apply:
            # camera lane-line matching: move the alignment so the route sits where the lane really is
            e = float(np.median([v[0] for v in self.lane_err]))
            nl = np.mean([v[1] for v in self.lane_err], axis=0)
            self.lane_err = []
            self.lane_corr = self.lane_corr - 0.3 * e * nl
            n = float(np.hypot(*self.lane_corr))
            if n > 2.5:
                self.lane_corr *= 2.5 / n
        mx, my = mx + self.lane_corr[0], my + self.lane_corr[1]
        ox, oy = self.offset
        nx, ny = ox + 0.3 * (mx - ox), oy + 0.3 * (my - oy)
        self.offset = (nx, ny)
        # route re-published in small 0.1 m steps (0.25 m jumps kicked the steering: zig-zag);
        # the heavy map display only every 0.5 m
        last = getattr(self, 'published_offset', (ox, oy))
        if math.hypot(nx - last[0], ny - last[1]) > 0.1:
            self.published_offset = (nx, ny)
            if self.path_world is not None:
                self.publish_route(self.path_world)
        last_map = getattr(self, 'published_map_offset', (ox, oy))
        if math.hypot(nx - last_map[0], ny - last_map[1]) > 0.5:
            self.published_map_offset = (nx, ny)
            self.publish_map()

    def on_goal(self, m):
        if self.offset is None:
            self.get_logger().warn('goal received before map alignment; retrying when aligned')
        self.goal_world = (m.pose.position.x + (self.offset or (0, 0))[0], m.pose.position.y + (self.offset or (0, 0))[1])
        self.goal_odom = (m.pose.position.x, m.pose.position.y)
        self.goal_reach = 60.0                 # clicked goal: snap to a road within 60 m
        self.parking = None
        # a click ON a parking bay (within 3.5 m of its centre) = park in that bay
        gx, gy = self.goal_world
        hit = min(self.bays, key=lambda b: math.hypot(b['x'] - gx, b['y'] - gy)) if self.bays else None
        if hit is not None and self.offset is not None and math.hypot(hit['x'] - gx, hit['y'] - gy) < 3.5:
            key = (hit['lot'], hit['index'])
            taken = self.bay_taken(hit)
            bay = None if taken else self.choose_bay(hit['x'], hit['y'], only=key)
            if bay is None:
                self.get_logger().warn(f"parking: bay {key[0]}/{key[1]} is "
                                       f"{'TAKEN' if taken else 'not reachable'} - taking the nearest free bay")
                bay = self.choose_bay(hit['x'], hit['y'])
            if bay is not None:
                self.parking = bay
                self.park_dest = (hit['x'], hit['y'], f"bay {key[0]}/{key[1]}")
                ax, ay = bay['approach']
                self.goal_odom = (ax - self.offset[0], ay - self.offset[1])
                self.goal_world = (ax, ay)
                self.goal_reach = 10.0
                self.get_logger().info(f"parking: you selected bay {bay['lot']}/{bay['index']} ({bay['mode']})")
            self.publish_bays()
        self.plan()

    def bay_taken(self, b):
        key = (b['lot'], b['index'])
        if key in self.bay_live:                 # seen by the lidar: overrides the map
            return self.bay_live[key]
        return bool(b.get('occupied')) or key in self.bays_excluded

    def on_occupancy(self, m):
        d = json.loads(m.data)
        key = (d['lot'], d['index'])
        if self.bay_live.get(key) == d['occupied']:
            return
        self.bay_live[key] = d['occupied']
        self.get_logger().info(f"parking: bay {key[0]}/{key[1]} seen {'TAKEN' if d['occupied'] else 'FREE'}")
        self.publish_bays()
        if d['occupied'] and self.parking is not None and (self.parking['lot'], self.parking['index']) == key:
            # the bay we were heading for has just been taken: nearest free one near the destination
            self.on_parking_next(String(data=json.dumps({'lot': key[0], 'index': key[1]})))

    def publish_bays(self):
        if not self.bays or self.offset is None:
            return
        self.pub_bay_data.publish(String(data=json.dumps([
            {'lot': b['lot'], 'index': b['index'], 'x': b['x'] - self.offset[0], 'y': b['y'] - self.offset[1],
             'yaw': b['yaw'], 'length': b['length'], 'width': b['width']} for b in self.bays])))
        ma = MarkerArray()
        sel = (self.parking['lot'], self.parking['index']) if self.parking else None
        stamp = self.get_clock().now().to_msg()
        for k, b in enumerate(self.bays):
            key = (b['lot'], b['index'])
            taken = self.bay_taken(b)
            col = (0.1, 0.4, 1.0) if key == sel else ((0.9, 0.15, 0.1) if taken else (0.1, 0.85, 0.2))
            yaw = b['yaw']
            for kind in ('box', 'text'):
                mk = Marker()
                mk.header.frame_id, mk.header.stamp = 'odom', stamp
                mk.ns, mk.id = f'bays_{kind}', k
                mk.pose.position.x = b['x'] - self.offset[0]
                mk.pose.position.y = b['y'] - self.offset[1]
                mk.pose.orientation.z, mk.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
                mk.color.r, mk.color.g, mk.color.b, mk.color.a = col[0], col[1], col[2], 0.55 if kind == 'box' else 1.0
                if kind == 'box':
                    mk.type = Marker.CUBE
                    mk.scale.x, mk.scale.y, mk.scale.z = float(b['length']) - 0.2, float(b['width']) - 0.2, 0.05
                    mk.pose.position.z = 0.05
                else:
                    mk.type = Marker.TEXT_VIEW_FACING
                    mk.scale.z = 0.9
                    mk.pose.position.z = 1.5
                    mk.text = f"P{b['lot']}-{b['index']} " + ('SELECTED' if key == sel else ('TAKEN' if taken else 'FREE'))
                ma.markers.append(mk)
        self.pub_bays.publish(ma)

    def on_go_to(self, m):
        q = m.data.strip().lower()
        emergency = q.startswith('emergency:')
        if emergency:
            q = q.split(':', 1)[1].strip()
        if self.offset is None:
            self.get_logger().warn('go_to before map alignment: ignored')
            return
        hit = self.nearest_place(q)
        if hit is None:                       # by name: the place's nearest reachable road point
            named = [p for p in self.places if p['name'].lower() == q]
            if named and self.pose is not None:
                here = (self.pose[0] + self.offset[0], self.pose[1] + self.offset[1])
                path = self.map.route_near(here, self.pose[2], (named[0]['x'], named[0]['y']), max_distance=200.0)
                if path is not None:
                    hit = (named[0]['name'], named[0]['x'], named[0]['y'],
                           float(np.hypot(*np.diff(path, axis=0).T).sum()))
        if hit is None:
            self.get_logger().warn(f'go_to {q!r}: no such place on the map')
            self.pub_dest.publish(String(data=json.dumps({'query': q, 'found': False})))
            return
        name, x, y, L = hit
        self.goal_odom = (x - self.offset[0], y - self.offset[1])
        self.goal_world = (x, y)
        self.goal_reach = 200.0                # places: drive to their nearest reachable road point
        self.parking = None
        if not emergency and self.driver_state != 'EMERGENCY':
            bay = self.choose_bay(x, y)
            self.park_dest = (x, y, name)
            if bay is not None:
                self.parking = bay
                ax, ay = bay['approach']
                self.goal_odom = (ax - self.offset[0], ay - self.offset[1])
                self.goal_world = (ax, ay)
                self.goal_reach = 10.0
                self.get_logger().info(f"parking: bay {bay['lot']}/{bay['index']} ({bay['mode']}) "
                                       f"{bay['dist']:.0f} m from {name}")
        self.get_logger().info(f'go_to {q!r}: {name} at ({x:.0f}, {y:.0f}), {L:.0f} m by road')
        self.pub_dest.publish(String(data=json.dumps({'query': q, 'found': True, 'name': name,
                                                      'x': x, 'y': y, 'route_length': round(L, 1)})))
        self.plan()

    def plan(self):
        if self.map is None or self.offset is None or self.pose is None or self.goal_world is None:
            return
        self.goal_world = (self.goal_odom[0] + self.offset[0], self.goal_odom[1] + self.offset[1])
        start = (self.pose[0] + self.offset[0], self.pose[1] + self.offset[1])
        path = self.map.route_near(start, self.pose[2], self.goal_world, max_distance=getattr(self, 'goal_reach', 60.0))
        if path is None:
            self.get_logger().warn('no route to goal')
            self.publish_status('no_route')
            return
        self.path_world = path
        self.publish_route(path)
        length = float(np.hypot(*np.diff(path, axis=0).T).sum())
        self.get_logger().info(f'route: {len(path)} points, {length:.0f} m')
        self.publish_status('route', length)

    def publish_route(self, path):
        msg = Path()
        msg.header.frame_id = 'odom'
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y in path:
            p = PoseStamped()
            p.header = msg.header
            p.pose.position.x, p.pose.position.y = float(x - self.offset[0]), float(y - self.offset[1])
            p.pose.orientation.w = 1.0
            msg.poses.append(p)
        self.pub_path.publish(msg)
        lines = PoseArray(header=msg.header)
        for x, y, h, speed, signal in self.map.last_stop_lines:   # junction entries on this route
            q = Pose()
            # z = speed limit; NEGATIVE z marks a junction known to have NO traffic light (signal map)
            z = speed if signal is not False else -speed
            q.position.x, q.position.y, q.position.z = x - self.offset[0], y - self.offset[1], z
            q.orientation.z, q.orientation.w = math.sin(h / 2), math.cos(h / 2)
            lines.poses.append(q)
        self.pub_lines.publish(lines)
        self.pub_junctions.publish(String(data=json.dumps([
            {'x': J['x'] - self.offset[0], 'y': J['y'] - self.offset[1], 'minor': J['minor'], 'state': J['state']}
            for J in getattr(self.map, 'last_junctions', [])])))
        from sensor_msgs_py import point_cloud2
        w = self.map.left_boundary(path)
        pts = [(float(x - self.offset[0]), float(y - self.offset[1]), 0.5) for x, y in w]
        self.pub_walls.publish(point_cloud2.create_cloud_xyz32(msg.header, pts))
        self.publish_markers(msg)

    def report_road(self):
        if self.map is None or self.offset is None or self.pose is None:
            return
        here = (self.pose[0] + self.offset[0], self.pose[1] + self.offset[1])
        info = self.map.road_info(here, self.pose[2])
        if info is None:
            return
        self.pub_road.publish(String(data=json.dumps(info)))
        key = (info['type'], info['name'], info['lanes'])
        if key != self.road_now and info['type'] != 'junction':
            self.road_now = key
            words = {'one_way': 'ONE-WAY road', 'two_way': 'TWO-WAY road', 'highway': 'HIGHWAY'}[info['type']]
            self.get_logger().info(f"now on: {words}{' ' + repr(info['name']) if info['name'] else ''}, "
                                   f"{info['lanes']} lane(s) our way (in lane {info['lane']}), "
                                   f"limit {info['speed_limit'] * 3.6:.0f} km/h")
        from std_msgs.msg import Header
        m = Marker(header=Header(frame_id='base_link', stamp=self.get_clock().now().to_msg()), ns='car', id=0,
                   type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
        m.pose.position.z, m.pose.orientation.w, m.scale.z = 4.0, 1.0, 1.6
        alarm = self.driver_state not in ('OK', None)
        m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.25, 0.2, 1.0) if alarm else (0.4, 1.0, 0.5, 1.0)
        road = {'one_way': 'one-way', 'two_way': 'two-way', 'highway': 'HIGHWAY', 'junction': 'junction'}[info['type']]
        m.text = (f"{road} {info['name']}  {info['speed_limit'] * 3.6:.0f} km/h"
                  + (f"\nDRIVER: {self.driver_state}" if alarm else ''))
        self.pub_label.publish(m)

    def check(self):
        self.realign()
        self.report_road()
        if self.goal_world is not None and self.path_world is None:
            self.plan()                               # goal arrived before alignment
            return
        if self.path_world is None or self.pose is None or self.offset is None:
            return
        here = np.array([self.pose[0] + self.offset[0], self.pose[1] + self.offset[1]])
        d = float(np.min(np.hypot(*(self.path_world - here).T)))
        reach = 2.0 if self.parking is not None else 4.0
        if np.hypot(*(self.path_world[-1] - here)) < reach:
            self.get_logger().info('goal reached: route cleared')
            if self.parking is not None:
                b = self.parking
                self.pub_parking.publish(String(data=json.dumps({
                    'mode': b['mode'], 'x': b['x'] - self.offset[0], 'y': b['y'] - self.offset[1], 'yaw': b['yaw'],
                    'lot': b['lot'], 'index': b['index']})))
                self.get_logger().info(f"parking manoeuvre: {b['mode']} into bay {b['lot']}/{b['index']}")
                self.parking = None
            self.path_world = None
            self.goal_world = None
            self.pub_path.publish(Path(header=self.make_header()))
            from sensor_msgs_py import point_cloud2
            self.pub_walls.publish(point_cloud2.create_cloud_xyz32(self.make_header(), []))
            self.publish_status('goal_reached')
        elif d > self.replan_distance:
            self.get_logger().info(f'{d:.1f} m off the route: re-planning')
            self.plan()

    def make_header(self):
        from std_msgs.msg import Header
        return Header(frame_id='odom', stamp=self.get_clock().now().to_msg())

    def publish_markers(self, path_msg):
        from geometry_msgs.msg import Point
        ma = MarkerArray()
        m = Marker(header=path_msg.header, ns='route', id=0, type=Marker.LINE_STRIP, action=Marker.ADD)
        m.scale.x = 0.6
        m.color.r, m.color.g, m.color.b, m.color.a = 0.2, 0.6, 1.0, 0.9
        m.pose.orientation.w = 1.0
        m.points = [Point(x=p.pose.position.x, y=p.pose.position.y, z=0.1) for p in path_msg.poses]
        ma.markers.append(m)
        self.pub_markers.publish(ma)

    def publish_signals(self):
        if self.offset is None:
            return
        from std_msgs.msg import Header
        pa = PoseArray(header=Header(frame_id='odom', stamp=self.get_clock().now().to_msg()))
        for x, y, yaw in self.lights:
            q = Pose()
            q.position.x, q.position.y, q.position.z = x - self.offset[0], y - self.offset[1], 3.0
            q.orientation.z, q.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            pa.poses.append(q)
        self.pub_signals.publish(pa)
        self.pub_spat.publish(String(data=json.dumps([
            {'x': L['x'] - self.offset[0], 'y': L['y'] - self.offset[1], 'yaw': L.get('yaw', 0.0),
             'green': L.get('green', 60.0), 'red': L.get('red', 15.0), 'start_green': L.get('start_green', True)}
            for L in self.light_info])))

    def publish_map(self):
        """The whole lane map in odom (latched), so a destination can be clicked in RViz with
        '2D Goal Pose': road edges as wide grey ribbons, lane centres with direction arrows,
        junction connections thin, signalised stop lines red."""
        if self.map is None or self.offset is None:
            return
        from geometry_msgs.msg import Point
        from std_msgs.msg import Header
        hdr = Header(frame_id='odom', stamp=self.get_clock().now().to_msg())
        ox, oy = self.offset
        ma = MarkerArray()

        def marker(mid, ns, kind, scale, rgba):
            m = Marker(header=hdr, ns=ns, id=mid, type=kind, action=Marker.ADD)
            m.scale.x = scale
            m.color.r, m.color.g, m.color.b, m.color.a = rgba
            m.pose.orientation.w = 1.0
            return m
        road = marker(0, 'road', Marker.LINE_LIST, 3.2, (0.32, 0.32, 0.34, 1.0))       # two-way: grey
        road_hw = marker(4, 'road_highway', Marker.LINE_LIST, 3.4, (0.15, 0.3, 0.6, 1.0))  # highway: blue
        road_ow = marker(5, 'road_one_way', Marker.LINE_LIST, 3.2, (0.55, 0.35, 0.1, 1.0)) # one-way: orange
        types = self.map.road_types()
        centre = marker(1, 'lane_centres', Marker.LINE_LIST, 0.12, (0.9, 0.9, 0.9, 0.8))
        junction = marker(2, 'junctions', Marker.LINE_LIST, 0.08, (0.6, 0.6, 0.65, 0.6))
        arrows = marker(3, 'lane_direction', Marker.LINE_LIST, 0.15, (1.0, 0.85, 0.2, 0.9))
        for lane in self.map.lanes.values():
            pts = lane.pts
            for a, b in zip(pts[:-1], pts[1:]):
                pa, pb = Point(x=float(a[0] - ox), y=float(a[1] - oy), z=-0.3), Point(x=float(b[0] - ox), y=float(b[1] - oy), z=-0.3)
                if lane.internal:
                    junction.points += [pa, pb]
                else:
                    rm = {'highway': road_hw, 'one_way': road_ow}.get(types.get(lane.edge), road)
                    rm.points += [Point(x=pa.x, y=pa.y, z=-0.35), Point(x=pb.x, y=pb.y, z=-0.35)]
                    centre.points += [pa, pb]
            if not lane.internal and lane.length > 8:                     # one chevron per lane
                m_ = lane.at(lane.length / 2)
                d = lane.at(lane.length / 2 + 1.0) - m_
                d = d / max(np.hypot(*d), 1e-6)
                n_ = np.array([-d[1], d[0]])
                tip = m_ + 1.2 * d
                for side in (1, -1):
                    tail = m_ - 0.6 * d + side * 0.8 * n_
                    arrows.points += [Point(x=float(tail[0] - ox), y=float(tail[1] - oy), z=-0.25),
                                      Point(x=float(tip[0] - ox), y=float(tip[1] - oy), z=-0.25)]
        ma.markers += [m_ for m_ in (road, road_hw, road_ow) if m_.points] + [centre, junction, arrows]
        ma.markers += self.place_markers(hdr, ox, oy)
        self.pub_map.publish(ma)
        self.publish_signals()
        from collections import Counter
        kinds = Counter(types.get(l.edge) for l in self.map.road_lanes)
        self.get_logger().info(f'road types (lanes): {dict(kinds)}')
        self.get_logger().info(f'lane map published on /map/lanes ({len(self.map.road_lanes)} lanes): '
                               f"click a destination in RViz with '2D Goal Pose'")

    COLOURS = {'hospital': (0.95, 0.15, 0.15), 'restaurant': (1.0, 0.6, 0.1), 'shop': (0.95, 0.75, 0.2),
               'museum': (0.6, 0.4, 0.9), 'landmark': (0.4, 0.7, 1.0), 'bus_stop': (0.2, 0.8, 0.9),
               'toilet': (0.5, 0.8, 0.5), 'house': (0.75, 0.68, 0.6), 'residential': (0.7, 0.6, 0.5),
               'building': (0.65, 0.65, 0.7), 'construction': (0.8, 0.7, 0.3)}

    def place_markers(self, hdr, ox, oy):
        """Buildings as extruded footprints coloured by kind, with a name label above each place
        that matters for navigation (hospital, shops, restaurants, bus stops, landmarks)."""
        from geometry_msgs.msg import Point
        out = []
        for i, p in enumerate(self.places):
            r, g, b = self.COLOURS.get(p['kind'], (0.65, 0.65, 0.7))
            foot = [(x - ox, y - oy) for x, y in p['footprint']]
            h = min(float(p.get('height', 6.0)), 25.0) * 0.3          # flattened so labels stay readable
            m = Marker(header=hdr, ns='buildings', id=i, type=Marker.TRIANGLE_LIST, action=Marker.ADD)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 1.0
            m.color.r, m.color.g, m.color.b, m.color.a = r, g, b, 0.85
            cx, cy = sum(x for x, _ in foot) / len(foot), sum(y for _, y in foot) / len(foot)
            for (x0, y0), (x1, y1) in zip(foot, foot[1:] + foot[:1]):
                m.points += [Point(x=cx, y=cy, z=h), Point(x=x0, y=y0, z=h), Point(x=x1, y=y1, z=h)]       # roof
                m.points += [Point(x=x0, y=y0, z=0.0), Point(x=x1, y=y1, z=0.0), Point(x=x1, y=y1, z=h),   # wall
                             Point(x=x0, y=y0, z=0.0), Point(x=x1, y=y1, z=h), Point(x=x0, y=y0, z=h)]
            out.append(m)
            if p['kind'] in ('house', 'building', 'residential'):
                continue
            t = Marker(header=hdr, ns='place_names', id=i, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
            t.pose.position.x, t.pose.position.y, t.pose.position.z = p['x'] - ox, p['y'] - oy, h + 4.0
            t.pose.orientation.w = 1.0
            t.scale.z = 6.0 if p['kind'] == 'hospital' else 3.5
            t.color.r, t.color.g, t.color.b, t.color.a = (1.0, 0.3, 0.3, 1.0) if p['kind'] == 'hospital' else (1.0, 1.0, 1.0, 0.95)
            t.text = ('+ ' + p['name'] + ' +') if p['kind'] == 'hospital' else p['name']
            out.append(t)
        return out

    def on_parking_next(self, m):
        """The planner found the chosen bay occupied on arrival: exclude it, take the next free one."""
        d = json.loads(m.data)
        self.bays_excluded.add((d.get('lot'), d.get('index')))
        if self.park_dest is None:
            return
        x, y, name = self.park_dest
        bay = self.choose_bay(x, y)
        if bay is None:
            self.get_logger().warn('parking: no free bay left near the destination - stopping here')
            return
        self.parking = bay
        ax, ay = bay['approach']
        self.goal_odom = (ax - self.offset[0], ay - self.offset[1])
        self.goal_world = (ax, ay)
        self.goal_reach = 10.0
        self.get_logger().info(f"parking: bay {d.get('lot')}/{d.get('index')} occupied -> bay {bay['lot']}/{bay['index']} ({bay['mode']})")
        self.plan()

    def choose_bay(self, x, y, max_dist=150.0, only=None):
        """Nearest parking bay to the destination (x, y) and how to get in:
          'drive_in' - the bay lies ON a car-park lane: drive to it and stop inside;
          'parallel' - the bay is beside a road lane (~3.3 m right): stop 2R sin(theta) past it and
                       reverse in along two arcs (R = minimum turning radius 6.2 m).
        approach = where base_link (rear axle) must stop before the manoeuvre (world frame)."""
        if not self.bays or self.map is None:
            return None
        R = 2.995 / math.tan(0.45)
        best = None
        here = None if self.pose is None or self.offset is None else \
            (self.pose[0] + self.offset[0], self.pose[1] + self.offset[1])

        def reachable(pt):
            # the route must really END at the approach point (a car-park lane that is not linked
            # to the road network ended the route on the main road beside it: 'parked' 6.9 m off)
            if here is None:
                return True
            P = self.map.route_near(here, self.pose[2], pt, max_distance=2.0)
            return P is not None and math.hypot(P[-1][0] - pt[0], P[-1][1] - pt[1]) < 1.5
        for b in self.bays:
            dist = math.hypot(b['x'] - x, b['y'] - y)
            if dist > max_dist or self.bay_taken(b):
                continue                                       # taken (map / seen by the lidar)
            if only is not None and (b['lot'], b['index']) != only:
                continue                                       # the bay the user selected
            hits = []
            for yaw in (b['yaw'], b['yaw'] + math.pi):
                yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                for lane in self.map.road_lanes:                   # every lane beside the bay
                    if np.hypot(*(lane.pts - [b['x'], b['y']]).T).min() > 60.0:
                        continue                                   # quick reject (far away)
                    s_, dl, _ = lane.project(np.array([b['x'], b['y']]))
                    if dl > 4.5:
                        continue                                   # distance to the lane LINE
                    q0, q1 = lane.at(s_), lane.at(s_ + 1.0)
                    hl = math.atan2(q1[1] - q0[1], q1[0] - q0[0])
                    if abs(math.atan2(math.sin(hl - yaw), math.cos(hl - yaw))) < math.radians(20):
                        hits.append((yaw, lane, s_, dl))
            for yaw, lane, s_, dl in hits:
                t = np.array([math.cos(yaw), math.sin(yaw)])
                nl = np.array([-t[1], t[0]])
                q = lane.at(s_)
                lat = float((np.array([b['x'], b['y']]) - q) @ nl)          # + = bay left of the lane
                if abs(lat) < 1.0:
                    if lane.width >= 3.0:
                        continue                  # a normal road lane through the bay: never park on a road
                    mode, approach = 'drive_in', np.array([b['x'], b['y']]) - 1.45 * t
                elif lat < 0 and abs(lat) < 2 * R:                          # right of the lane (keep right)
                    th = math.acos(1.0 - abs(lat) / (2 * R))
                    D = 2 * R * math.sin(th)
                    approach = q + (D - 1.45) * t
                    mode = 'parallel'
                else:
                    continue
                cand = {'lot': b['lot'], 'index': b['index'], 'x': b['x'], 'y': b['y'], 'yaw': yaw, 'mode': mode,
                        'approach': (float(approach[0]), float(approach[1])), 'dist': dist}
                # PARALLEL from the road beside the bay is preferred: a 'drive_in' along the bay strip
                # drives through the other bays of the row - with cars parked in them the car stopped
                # at the car-park entrance (a parked car's roof 1.5 m ahead)
                rank = (0 if mode == 'parallel' else 1, dist)
                if (best is None or rank < best['rank']) and reachable(cand['approach']):
                    cand['rank'] = rank
                    best = cand
        return best

    def nearest_place(self, kind):
        """(name, world x, y) of the nearest place of `kind` by ROUTE length, or None."""
        if self.pose is None or self.offset is None or self.map is None:
            return None
        here = (self.pose[0] + self.offset[0], self.pose[1] + self.offset[1])
        best = None
        for p in self.places:
            if p['kind'] != kind:
                continue
            path = self.map.route_near(here, self.pose[2], (p['x'], p['y']), max_distance=200.0)
            if path is None:
                continue
            L = float(np.hypot(*np.diff(path, axis=0).T).sum())
            if best is None or L < best[3]:
                best = (p['name'], p['x'], p['y'], L)
        return best

    def publish_status(self, state, length=0.0):
        self.pub_status.publish(String(data=json.dumps({'state': state, 'length': round(length, 1)})))


def main():
    rclpy.init()
    node = RoutePlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
