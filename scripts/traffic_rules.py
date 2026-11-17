#!/usr/bin/env python3
"""Traffic rules: stop lines from the lane map + signals from the camera (Autoware-style split).

Input : /route/stop_lines (PoseArray, odom: where the route enters a junction, from the lane map)
        /odometry/filtered, /camera/image_raw, /perception/detections (RT-DETR boxes)
        /perception/tracks_labeled (lidar tracks with camera labels)
Output: /behaviour/stop_points  PointCloud2 (base_link): a virtual wall across the lane at the stop
                                line when the car must stop. Used by the planner AND the safety
                                shield (a red light is a hard rule, not a preference).
        /behaviour/speed_limit  Float32 (m/s): lane speed limit, lowered near pedestrians
        /behaviour/state        String JSON: light colour, stop sign, distance to stop line, decision
        /behaviour/light_crop   Image: the traffic-light crop that decided the colour (debug)

Rules
  red light          stop at the line
  orange (yellow)    stop if the car can still stop comfortably before the line, else continue
  stop sign          stop at the line, wait `stop_wait` s, then continue (shield TTC covers cross traffic)
  pedestrian nearby  speed limit `ped_speed` within `ped_distance` of a tracked person
Light colour: the lit lamp inside a detected traffic-light box is the bright, saturated blob;
its hue gives red / orange / green. The state must be seen in `confirm` consecutive frames.
"""
import json
import math
from collections import deque

import numpy as np
import rclpy
from auto_car.msg import TrackArray
from geometry_msgs.msg import PoseArray
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32, Header, String
from vision_msgs.msg import Detection2DArray

from auto_car_core.decision import CREEP, GO, NAMES, YIELD, GiveWayDecider

CAR_FRONT = 4.05
CAM_X = 1.0              # front camera on the roof mount, 1.0 m ahead of base_link


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def lamp_colour(crop):
    """('red'|'orange'|'green'|None, pixel count) of the lit lamp in a BGR crop."""
    colour, count, _ = lamp(crop)
    return colour, count


def lamp(crop):
    """(colour, pixel count, shape) of the lit lamp; shape is 'arrow' or 'round'.

    The junction has separate ARROW signals for turning left. Seen in testing: a car going
    straight stopped for a red left arrow while its own round light was green. Shape: an arrow
    is a wide, partly filled blob; a round lamp is compact and nearly fills its box.
    """
    if crop.size == 0:
        return None, 0, None
    b, g, r = (crop[..., i].astype(np.float32) for i in range(3))
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = mx - mn + 1e-6
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 30
    s = d / (mx + 1e-6) * 255
    lit = (mx > 150) & (s > 100)
    counts = {'red': int((lit & ((h < 10) | (h > 165))).sum()),
              'orange': int((lit & (h >= 10) & (h < 35)).sum()),
              'green': int((lit & (h >= 40) & (h < 95)).sum())}
    best = max(counts, key=counts.get)
    if counts[best] < 2:
        return None, 0, None
    from scipy import ndimage
    hue = {'red': (h < 10) | (h > 165), 'orange': (h >= 10) & (h < 35), 'green': (h >= 40) & (h < 95)}[best]
    lab, n = ndimage.label(lit & hue)
    if n == 0:
        return best, counts[best], 'round'
    k = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    ys, xs = np.nonzero(lab == k)
    w, hh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    fill = len(xs) / float(w * hh)
    shape = 'arrow' if (w / hh > 1.4 or hh / w > 1.4 or fill < 0.55) and len(xs) >= 6 else 'round'
    return best, counts[best], shape


class TrafficRules(Node):
    def __init__(self):
        super().__init__('traffic_rules')
        p = self.declare_parameter
        self.lookahead = p('lookahead', 50.0).value
        self.lateral = p('stop_line_lateral', 8.0).value
        self.stop_wait = p('stop_wait', 2.0).value
        self.confirm = p('confirm', 3).value
        self.a_comfort = p('a_comfort', 3.0).value
        self.ped_speed = p('ped_speed', 3.0).value
        self.ped_distance = p('ped_distance', 25.0).value     # slow down when first seen, not when close
        self.speed_default = p('speed_limit', 13.9).value
        self.stop_lines = []        # [(x, y, heading, speed)] odom
        self.unsignalised = set()
        self.pose = None
        self.v = 0.0
        self.images = deque(maxlen=8)
        self.light_hist = deque(maxlen=self.confirm)
        self.light = None
        self.stop_sign_seen = 0.0
        self.served = set()         # stop lines already handled for a stop sign
        self.wait_start = None
        self.people = []
        # Latched signal per stop line: once red/orange is seen while approaching, keep stopping until
        # green is SEEN. Close to the line the light (overhead) leaves the camera view; 'not visible'
        # must never mean 'go' (found in testing: the car drove on with light=None).
        self.latched = {}           # stop line index -> (colour, time)
        self.latch_timeout = self.declare_parameter('latch_timeout', 60.0).value
        self.yield_gap = self.declare_parameter('pedestrian_yield_gap', 2.5).value   # m bumper -> person
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(PoseArray, '/route/stop_lines', self.on_lines, latched)
        # mapped light positions (odom) + camera model: read OUR light where the map says it is
        self.signal_lights = []
        self.spat = []
        self.create_subscription(String, '/route/signal_schedule', lambda m: setattr(self, 'spat', json.loads(m.data)), latched)
        self.spat_used = False
        # right of way per stop line (same order as /route/stop_lines): give way on minor approaches
        self.junctions = []
        self.create_subscription(String, '/route/junctions', lambda m: setattr(self, 'junctions', json.loads(m.data)), latched)
        self.moving = []
        # RL decision layer for give-way points (always on; rule fallback if the model is missing)
        self.decider = GiveWayDecider(self.declare_parameter('giveway_policy', '').value or
                                      __import__('auto_car_core.decision', fromlist=['x']).DEFAULT_MODEL)
        self.get_logger().info(f'give-way decisions: {self.decider.source}')
        self.gw_wait_start = None
        self.decision = None
        self.creep = False
        self.create_subscription(TrackArray, '/perception/tracks', lambda m: setattr(self, 'moving', [
            (t.position.x, t.position.y, t.velocity.x, t.velocity.y) for t in m.tracks if t.moving]), 10)
        from std_msgs.msg import String as _S
        self.pub_indicator = self.create_publisher(_S, '/vehicle/indicator', 5)
        self.indicator = 'off'
        self.create_subscription(PoseArray, '/route/signal_lights', lambda m: setattr(
            self, 'signal_lights', [(p.position.x, p.position.y, yaw_of(p.orientation)) for p in m.poses]), latched)
        from sensor_msgs.msg import CameraInfo
        self.cam_info = None
        self.create_subscription(CameraInfo, '/camera/camera_info', lambda m: setattr(self, 'cam_info', m), 2)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 20)
        self.route = None
        self.lamps_seen = []
        self.create_subscription(Path, '/route/path', lambda m: setattr(self, 'route', np.array(
            [[p.pose.position.x, p.pose.position.y] for p in m.poses]) if len(m.poses) > 1 else None), latched)
        self.create_subscription(Image, '/camera/image_raw', lambda m: self.images.append(m), 4)
        self.create_subscription(Detection2DArray, '/perception/detections', self.on_detections, 10)
        self.create_subscription(TrackArray, '/perception/tracks_labeled', self.on_tracks, 10)
        self.pub_stop = self.create_publisher(PointCloud2, '/behaviour/stop_points', 5)
        self.pub_speed = self.create_publisher(Float32, '/behaviour/speed_limit', 5)
        # Distance bumper -> stop line when the car must stop (1e9 = no stop). A stop rule is NOT an
        # obstacle: walls can be steered around (seen in testing), a distance cannot.
        self.pub_stop_dist = self.create_publisher(Float32, '/behaviour/stop_distance', 5)
        self.light_lines = {}       # stop line index -> last time a traffic light was seen for it
        self.pub_state = self.create_publisher(String, '/behaviour/state', 5)
        self.road_limit = None
        self.create_subscription(String, '/route/road', lambda m: setattr(
            self, 'road_limit', json.loads(m.data).get('speed_limit')), 5)
        self.pub_crop = self.create_publisher(Image, '/behaviour/light_crop', 2)
        self.create_timer(0.1, self.loop)
        self.get_logger().info('traffic_rules: lights, stop signs, speed limits, pedestrians')

    # --------------------------------------------------------------- inputs
    def on_lines(self, m):
        # z < 0: junction known to have no traffic light (signal map) -> lights seen are not for it
        self.stop_lines = [(p.position.x, p.position.y, yaw_of(p.orientation), abs(p.position.z) or self.speed_default)
                           for p in m.poses]
        self.unsignalised = {i for i, p in enumerate(m.poses) if p.position.z < 0}
        self.served.clear()

    def on_odom(self, m):
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
        self.v = m.twist.twist.linear.x

    def _line_has_stop(self, idx):
        """HD-map association for a camera STOP sign: only on approaches the lane map marks as
        stop-controlled (SUMO link state 's'). The village has no stop signs at all, yet the detector
        'saw' one at every roundabout entry (a red speed-limit ring / red building / red car) and the
        car stopped there every time; give-way approaches are decided by the RL decision layer."""
        J = self.junctions[idx] if idx < len(self.junctions) else None
        return J is not None and J.get('state') == 's'

    def on_tracks(self, m):
        self.people = [(t.position.x, t.position.y) for t in m.tracks if t.label == 'person']
        self.people_v = [(t.position.x, t.position.y, t.velocity.x, t.velocity.y) for t in m.tracks if t.label == 'person']

    def on_detections(self, m):
        now = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        img = next((i for i in self.images if i.header.stamp == m.header.stamp), None)
        boxes = []
        for d in m.detections:
            cls = d.results[0].hypothesis.class_id
            cx, cy = d.bbox.center.position.x, d.bbox.center.position.y
            w, h = d.bbox.size_x, d.bbox.size_y
            if cls == 'stop sign' and w * h > 150:            # close enough to matter
                self.stop_sign_seen = now
            if cls == 'traffic light' and img is not None:
                boxes.append((cx, cy, w, h))
        expected = self.expected_light_u()
        if expected is not None and boxes:
            # Map-based association (as Autoware does): our approach's light is where the map puts it;
            # near the stop line it is at the image EDGE, which the old 'central 70 %' filter dropped,
            # leaving only cross-street lights in view (the car waited at a red arrow forever).
            near = [b for b in boxes if abs(b[0] - expected) < max(60.0, 1.5 * b[2])]
            boxes = near if near else [b for b in boxes if 0.15 * img.width < b[0] < 0.85 * img.width]
        elif img is not None:
            boxes = [b for b in boxes if 0.15 * img.width < b[0] < 0.85 * img.width]
        colour = None
        if boxes:
            frame = np.frombuffer(bytes(img.data), np.uint8).reshape(img.height, img.width, -1)[:, :, :3]
            seen = []
            for cx, cy, w, h in boxes:
                x0, x1 = int(max(cx - 0.6 * w, 0)), int(min(cx + 0.6 * w, img.width))
                y0, y1 = int(max(cy - 0.6 * h, 0)), int(min(cy + 0.6 * h, img.height))
                crop = np.ascontiguousarray(frame[y0:y1, x0:x1])
                c, cnt, shape = lamp(crop)
                if c:
                    seen.append((c, shape, cnt, crop))
            # Signals for OUR manoeuvre: arrows apply to left turns only
            turn = self.next_turn()
            relevant = [x for x in seen if x[1] == 'round']
            if turn == 'left' and any(x[1] == 'arrow' for x in seen):
                relevant = [x for x in seen if x[1] == 'arrow']
            if relevant:
                # OUR light faces us: its lit lamp is a full bright disk, the cross street's lights are
                # seen from the side as thin slivers. Follow the brightest lamp; if another lamp is
                # nearly as bright (within 30%), be cautious and take the more restrictive of the two.
                relevant.sort(key=lambda x: -x[2])
                top = relevant[0]
                close = [x for x in relevant if x[2] >= 0.7 * top[2]]
                order = {'red': 0, 'orange': 1, 'green': 2}
                chosen = min(close, key=lambda x: order[x[0]])
                colour, crop = chosen[0], chosen[3]
                self.pub_crop.publish(Image(header=img.header, height=crop.shape[0], width=crop.shape[1],
                                            encoding='bgr8', step=crop.shape[1] * 3, data=crop.tobytes()))
            self.lamps_seen = [(c, sh) for c, sh, _, _ in seen]
        self.light_hist.append(colour)
        if len(self.light_hist) == self.confirm and len(set(self.light_hist)) == 1:
            self.light = (colour, now)

    @staticmethod
    def spat_colour(L, t):
        """Colour of a Webots generic traffic light at sim time t (its exact schedule = V2I SPaT).
        Controller tick 0.512 s: every phase lasts a whole number of ticks."""
        tick = 0.512
        g = math.ceil(L['green'] / tick - 1e-9) * tick
        r = math.ceil(L['red'] / tick - 1e-9) * tick
        o = math.ceil(1.5 / tick - 1e-9) * tick
        if L['start_green']:
            phases = [('green', g), ('orange', o), ('red', r), ('red', o)]       # orange before green = wait
        else:
            phases = [('red', r), ('red', o), ('green', g), ('orange', o)]
        k = t % sum(d for _, d in phases)
        for col, d in phases:
            if k < d:
                return col
            k -= d
        return 'red'

    def spat_for_line(self, idx):
        """V2I: the scheduled colour of the light facing this approach, or None (no V2I data)."""
        if not self.spat or idx is None:
            return None
        lx, ly, lh, _ = self.stop_lines[idx]
        best = None
        for L in self.spat:
            along = (L['x'] - lx) * math.cos(lh) + (L['y'] - ly) * math.sin(lh)
            dyaw = abs(math.atan2(math.sin(L['yaw'] - lh - math.pi), math.cos(L['yaw'] - lh - math.pi)))
            d = math.hypot(L['x'] - lx, L['y'] - ly)
            if dyaw < math.radians(35) and -15.0 < along < 40.0 and d < 45.0 and (best is None or d < best[0]):
                best = (d, L)
        if best is None:
            return None
        return self.spat_colour(best[1], self.get_clock().now().nanoseconds * 1e-9)

    def expected_light_u(self):
        """Image column where the mapped light of the next stop line should appear, or None."""
        line = self.next_stop_line()
        if line is None or not self.signal_lights or self.cam_info is None or self.pose is None:
            return None
        idx = line[0]
        lx, ly, lh, _ = self.stop_lines[idx]
        # the mapped light that FACES this approach (far-side head, up to 40 m past the line)
        best = None
        for x, y, yaw in self.signal_lights:
            along = (x - lx) * math.cos(lh) + (y - ly) * math.sin(lh)
            dyaw = abs(math.atan2(math.sin(yaw - lh - math.pi), math.cos(yaw - lh - math.pi)))
            if dyaw < math.radians(35) and -15.0 < along < 40.0:
                d = math.hypot(x - lx, y - ly)
                if best is None or d < best[0]:
                    best = (d, x, y)
        if best is None:
            return None                                  # no mapped light for this approach
        _, x, y = best
        px, py, yaw = self.pose
        c, s_ = math.cos(yaw), math.sin(yaw)
        bx, by = c * (x - px) + s_ * (y - py), -s_ * (x - px) + c * (y - py)
        bx -= CAM_X
        if bx < 1.0:
            return None
        fx, cx = self.cam_info.k[0], self.cam_info.k[2]
        return cx - fx * by / bx

    # ---------------------------------------------------------------- logic
    def next_stop_line(self):
        if self.pose is None:
            return None
        x, y, yaw = self.pose
        c, s = math.cos(yaw), math.sin(yaw)
        best = None
        for i, (lx, ly, lh, sp) in enumerate(self.stop_lines):
            dx, dy = lx - x, ly - y
            bx, by = c * dx + s * dy, -s * dx + c * dy
            if 0.0 < bx < self.lookahead and abs(by) < self.lateral and \
                    abs(math.atan2(math.sin(lh - yaw), math.cos(lh - yaw))) < math.radians(45):
                if best is None or bx < best[1]:
                    best = (i, bx, by, lh - yaw, sp)
        return best

    def next_turn(self):
        """'left' / 'straight' / 'right' for the next junction on the route (from the route path)."""
        line = self.next_stop_line()
        if line is None or self.route is None:
            return 'straight'
        lx, ly, lh, _ = self.stop_lines[line[0]]
        d = np.hypot(self.route[:, 0] - lx, self.route[:, 1] - ly)
        i = int(np.argmin(d))
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.route[i:], axis=0).T))]
        j = i + int(np.searchsorted(s, 20.0))
        if j >= len(self.route) - 1:
            return 'straight'
        v = self.route[j + 1] - self.route[j]
        turn = math.degrees(math.atan2(math.sin(math.atan2(v[1], v[0]) - lh), math.cos(math.atan2(v[1], v[0]) - lh)))
        return 'left' if turn > 35 else ('right' if turn < -35 else 'straight')

    def loop(self):
        now = self.get_clock().now().nanoseconds * 1e-9
        line = self.next_stop_line()
        light = self.light[0] if self.light and now - self.light[1] < 1.0 else None
        stop, reason = False, 'go'
        self.creep = False
        d_stop = None
        if line is not None and line[1] - CAR_FRONT < -0.5:
            line = None                                       # bumper past the line: committed, never stop in the junction
        if line is not None:
            idx, bx, by, rel, _ = line
            d_stop = bx - CAR_FRONT                           # bumper to line
            if idx in self.unsignalised:
                light = None                                  # no light at this junction (signal map)
            spat = self.spat_for_line(idx)
            if spat is not None:
                # V2I signal phase (SPaT) of OUR light: exact, independent of what else the camera sees
                # (a red 'lane closed' X sign was read as a permanent red light)
                light = spat
                if not self.spat_used:
                    self.spat_used = True
                    self.get_logger().info('traffic lights: using V2I signal timing (SPaT) where available, camera elsewhere')
            if light is not None:
                self.light_lines[idx] = now
            if light in ('red', 'orange'):
                self.latched[idx] = (light, now)
            elif light == 'green':
                self.latched.pop(idx, None)
            elif idx in self.latched and now - self.latched[idx][1] < self.latch_timeout:
                light = self.latched[idx][0]                  # not visible: keep the last red/orange
            if light == 'red':
                stop, reason = True, 'red_light'
            elif light == 'orange':
                need = self.v * self.v / (2 * self.a_comfort)
                stop, reason = (True, 'orange_light') if d_stop > need else (False, 'orange_too_late_to_stop')
            elif (now - self.stop_sign_seen < 3.0 and idx not in self.served and d_stop < 30.0
                  and now - self.light_lines.get(idx, -1e9) > 20.0     # signalised junction: lights rule
                  and self._line_has_stop(idx)):
                # map association: a camera stop sign only counts where the map has a stop line
                # 'stopped at the line': the planner halts 1-2 m before it (stop 1 m short + shield
                # margin); a 1.5 m zone was never reached and the car waited at the sign forever
                if d_stop < 3.0 and abs(self.v) < 0.15:
                    self.wait_start = self.wait_start or now
                    if now - self.wait_start >= self.stop_wait:
                        self.served.add(idx)
                        self.wait_start = None
                    else:
                        stop, reason = True, 'stop_sign_wait'
                else:
                    stop, reason = True, 'stop_sign'
            if not stop:
                self.wait_start = None if reason == 'go' else self.wait_start
        # Yield to people in or walking into our path (crosswalk or not): stop yield_gap short of them.
        # Path corridor = straight ahead, half width + 1.2 m; a person is 'entering' if their position
        # 2 s from now is inside it or they cross it. (The car used to pass a crossing person at 1.1 m.)
        if not stop:
            for px, py, vx, vy in getattr(self, 'people_v', []):
                ahead = px - CAR_FRONT
                if not (0.0 < ahead < 20.0):
                    continue
                y2 = py + 2.0 * vy
                half = 1.0 + 1.2
                if abs(py) < half or abs(y2) < half or py * y2 < 0:
                    d = ahead - self.yield_gap
                    if not stop or d < d_stop:
                        stop, reason, d_stop = True, 'yield_pedestrian', max(d, 0.0)
                        line = (None, px - self.yield_gap, 0.0, 0.0, line[4] if line is not None else self.speed_default)
        # GIVE WAY (yield sign / minor road / roundabout entry, no traffic light): stop at the line
        # while a moving vehicle would reach the junction within 4 s; go when there is a gap.
        if not stop and line is not None and line[0] is not None and light is None:
            idx = line[0]
            J = self.junctions[idx] if idx < len(self.junctions) else None
            if J is not None and J.get('minor') and self.pose is not None and d_stop is not None and d_stop > -0.5:
                px, py, yaw = self.pose
                c, s_ = math.cos(yaw), math.sin(yaw)
                jx, jy = c * (J['x'] - px) + s_ * (J['y'] - py), -s_ * (J['x'] - px) + c * (J['y'] - py)
                # RL DECISION LAYER (auto_car_core/decision.py, PPO from rl/train_decision.py):
                # YIELD / CREEP / GO at the line, wrapped by hard rules (committed past the line,
                # never GO with a priority vehicle < 1.5 s away); falls back to the 4 s gap rule.
                if d_stop < 25.0:
                    if abs(self.v) < 0.2 and d_stop < 3.0:
                        self.gw_wait_start = self.gw_wait_start or now
                    waited = now - self.gw_wait_start if self.gw_wait_start else 0.0
                    act, why = self.decider.decide(d_stop, self.v, waited, self.moving, jx, jy)
                    self.decision = {'action': NAMES[act], 'by': why, 'source': self.decider.source,
                                     'waited': round(waited, 1)}
                    if act == YIELD:
                        stop, reason = True, 'give_way'
                    elif act == CREEP:
                        stop, reason = True, 'give_way_creep'          # stop at the line, approach slowly
                        self.creep = True
                    else:
                        self.gw_wait_start = None
        if reason not in ('give_way', 'give_way_creep'):
            self.gw_wait_start = None if d_stop is None or d_stop > 3.0 or d_stop < -0.5 else self.gw_wait_start
        # INDICATOR: blink toward the turn within 30 m of the junction, off otherwise
        turn = self.next_turn()
        want = turn if (turn in ('left', 'right') and d_stop is not None and -2.0 < d_stop < 30.0) else 'off'
        if want != self.indicator:
            self.indicator = want
            from std_msgs.msg import String as _S
            self.pub_indicator.publish(_S(data=want))
        pts = np.zeros((0, 3), np.float32)
        if stop and line is not None:
            _, bx, by, rel, _ = line
            # wall across the lane, perpendicular to the lane direction, at the stop line
            n = np.array([-math.sin(rel), math.cos(rel)])
            ts = np.linspace(-15.0, 15.0, 61)     # wide: no arc may steer around a red light
            pts = np.c_[bx + ts * n[0], by + ts * n[1], np.full(len(ts), 0.5)].astype(np.float32)
        self.pub_stop_dist.publish(Float32(data=float(d_stop) if stop and d_stop is not None else 1e9))
        self.pub_stop.publish(point_cloud2.create_cloud_xyz32(Header(frame_id='base_link',
                                                                       stamp=self.get_clock().now().to_msg()), pts))
        # speed limit: the road we are on (lane map), else the next junction approach, else default
        limit = self.road_limit if self.road_limit else (line[4] if line is not None else self.speed_default)
        near_person = any(math.hypot(px, py) < self.ped_distance for px, py in self.people)
        if near_person:
            limit = min(limit, self.ped_speed)
        if self.creep:
            limit = min(limit, 1.5)                      # CREEP: edge up to the line at walking pace
        self.pub_speed.publish(Float32(data=float(limit)))
        self.pub_state.publish(String(data=json.dumps({
            'light': light, 'turn': self.next_turn(), 'lamps': self.lamps_seen, 'stop_sign': now - self.stop_sign_seen < 3.0, 'stop': stop, 'reason': reason,
            'stop_line_distance': None if d_stop is None else round(d_stop, 1), 'speed_limit': round(limit, 1),
            'pedestrian_near': near_person, 'decision': self.decision})))


def main():
    rclpy.init()
    node = TrafficRules()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
