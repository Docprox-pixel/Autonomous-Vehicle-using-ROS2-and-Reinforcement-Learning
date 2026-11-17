#!/usr/bin/env python3
"""Detect and track objects around the car; separate the dynamic world from the static one.

Input : /perception/obstacle_points  (base_link, road already removed by cloud_preprocess)
        /odometry/filtered            (EKF) ego pose, interpolated to each cloud's exact stamp
Output: /perception/tracks           auto_car/TrackArray, base_link: position, velocity, size, moving
        /perception/track_markers    MarkerArray for RViz (red = moving, grey = static object)
        /perception/static_points    obstacle points WITHOUT moving objects  -> OctoMap
        /scan_static                 static obstacles 0.25-2.2 m above the road, flattened -> SLAM

Pipeline per lidar frame (10 Hz):
  1. occupancy grid (0.4 m) of points up to 3 m above the road, 8-connected clusters
  2. object-sized clusters (<= 7 m long) become detections; long ones (walls, barriers) are static
  3. constant-velocity Kalman filter per track in the odom frame, greedy nearest-neighbour gating
  4. velocity is measured by REGISTERING the object's points between frames (2D ICP, translation),
     not from the box centre. While the ego car drives, the visible part of a long wall or a
     parked car changes and its centre "slides" at ego speed although every point is static;
     the registered points stay put, so the measured velocity is ~0.
  5. occupancy history (aperture problem): on a smooth wall the lidar hits themselves slide along
     the surface, so even registration can report motion for sparse far fragments. A world grid
     remembers when each cell was first seen occupied: a moving car occupies cells that were
     empty a moment ago, a wall fragment occupies cells already seen before.
  MOVING = confirmed AND speed > moving_speed AND registered displacement > min_displacement
           AND within moving_range AND most of its cells are new (first seen < history_age ago),
           held for moving_frames consecutive frames (hysteresis: one bad registration cannot flip
           the label either way).
Simulator-agnostic: uses only the sensor contract and the EKF.
"""
import math
import os
import time

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from auto_car.msg import Track, TrackArray
from collections import deque

from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy import ndimage
from scipy.spatial import cKDTree
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class KalmanTrack:
    _next_id = 1

    def __init__(self, z, size, t):
        self.id = KalmanTrack._next_id
        KalmanTrack._next_id += 1
        self.x = np.array([z[0], z[1], 0.0, 0.0])
        self.P = np.diag([0.3, 0.3, 4.0, 4.0])
        self.birth = np.array(z[:2])
        self.size = np.array(size)
        self.t = t
        self.age = self.hits = 1
        self.missed_time = 0.0
        self.pts = None            # last world-frame 2D points of this object (for registration)
        self.old, self.range = 1.0, 0.0
        self.votes = 0               # consecutive frames meeting the moving criteria
        self.moving = False
        self.reg_disp = np.zeros(2)  # accumulated registered displacement since birth

    def predict(self, t, q=2.0):
        dt = max(t - self.t, 1e-3)
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q * q
        self.t = t
        self.age += 1
        return dt

    def update(self, z, size, v_meas=None, r_pos=1.0, r_vel=0.6):
        """Position from the box centre (loose: it slides with the visible part), velocity from
        point registration (tight). Without a velocity measurement only position is used."""
        if v_meas is None:
            H, zz, R = np.eye(4)[:2], np.asarray(z[:2]), np.eye(2) * r_pos ** 2
        else:
            H = np.eye(4)
            zz = np.r_[np.asarray(z[:2]), v_meas]
            R = np.diag([r_pos ** 2, r_pos ** 2, r_vel ** 2, r_vel ** 2])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (zz - H @ self.x)
        self.P = (np.eye(4) - K @ H) @ self.P
        self.size = 0.7 * self.size + 0.3 * np.asarray(size)
        self.hits += 1
        self.missed_time = 0.0

    @property
    def speed(self):
        return float(math.hypot(self.x[2], self.x[3]))


def register(prev, cur, init, iters=6, inlier=0.6):
    """Translation t that moves `prev` points onto `cur` (2D ICP, translation only)."""
    tree = cKDTree(prev)
    t = np.asarray(init, dtype=float).copy()
    for _ in range(iters):
        d, j = tree.query(cur - t)
        ok = d < inlier
        if ok.sum() < 3:
            return None
        t += np.median((cur - t)[ok] - prev[j[ok]], axis=0)
    d, _ = tree.query(cur - t)
    return t if np.mean(d < inlier) > 0.5 else None


class ObjectTracker(Node):
    def __init__(self):
        super().__init__('object_tracker')
        layout = yaml.safe_load(open(self.declare_parameter('sensor_layout', os.path.join(
            get_package_share_directory('auto_car'), 'config', 'sensor_layout.yaml')).value))
        self.ground_z = float(layout['ground_z'])
        p = self.declare_parameter
        self.cell = p('cell', 0.4).value
        self.max_range = p('max_range', 50.0).value
        self.max_len = p('max_object_length', 7.0).value
        self.min_pts = p('min_points', 4).value
        self.min_h = p('min_object_height', 0.4).value
        self.gate = p('gate', 2.5).value
        self.confirm_hits = p('confirm_hits', 4).value
        self.max_missed = p('max_missed_time', 0.8).value
        self.moving_speed = p('moving_speed', 1.5).value
        self.min_disp = p('min_displacement', 2.0).value
        self.moving_frames = p('moving_frames', 5).value
        self.min_reg_pts = p('min_registration_points', 10).value
        self.moving_range = p('moving_range', 30.0).value   # dense zone: corner + roof lidars overlap
        self.history_age = p('history_age', 1.5).value      # s
        self.max_old_frac = p('max_old_fraction', 0.4).value
        self.hist_res = 0.4
        self.hist_half = 400.0                              # m, grid covers +/-400 m around start
        n_cells = int(2 * self.hist_half / self.hist_res)
        # occupied_since: start of the current continuous occupancy of each cell; resets when the cell
        # was not seen occupied for history_gap s (so a lane cell is 'new' again for the next car).
        self.first_seen = np.full((n_cells, n_cells), np.inf, dtype=np.float32)
        self.last_seen = np.full((n_cells, n_cells), -np.inf, dtype=np.float32)
        self.history_gap = p('history_gap', 1.0).value
        self.hist_origin = None
        self.scan_min_h, self.scan_max_h = p('scan_min_height', 0.25).value, p('scan_max_height', 2.2).value
        self.bins = p('scan_bins', 720).value

        self.tracks = []
        self.odom = deque(maxlen=300)        # (t, x, y, yaw, v, yaw_rate) from the EKF
        self.pose_fallbacks = 0
        self.timing = []
        self.last_t = -np.inf
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 50)
        self.pub_tracks = self.create_publisher(TrackArray, '/perception/tracks', 10)
        self.pub_markers = self.create_publisher(MarkerArray, '/perception/track_markers', 10)
        self.pub_static = self.create_publisher(PointCloud2, '/perception/static_points', 5)
        self.pub_scan = self.create_publisher(LaserScan, '/scan_static', 5)
        self.create_subscription(PointCloud2, '/perception/obstacle_points', self.on_cloud, 5)
        self.get_logger().info('object_tracker ready')

    # ------------------------------------------------------------ geometry
    def on_odom(self, m):
        self.odom.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.pose.position.x,
                          m.pose.pose.position.y, yaw_of(m.pose.pose.orientation),
                          m.twist.twist.linear.x, m.twist.twist.angular.z))

    def ego_pose(self, stamp, max_extrapolation=0.15):
        """Ego pose at the EXACT time of the cloud (interpolated, or extrapolated a few ms).

        At 13 m/s a 0.1 s mismatch shifts every point 1.3 m in the world and static objects
        appear to move. The EKF output lags the lidar by ~10-25 ms, so blocking on TF is not an
        option (with sim time it even deadlocks the node's clock).
        """
        t = stamp.sec + stamp.nanosec * 1e-9
        if not self.odom:
            return None
        if t >= self.odom[-1][0]:
            t0, x, y, yaw, v, w = self.odom[-1]
            dt = t - t0
            if dt > max_extrapolation:
                self.pose_fallbacks += 1
                if self.pose_fallbacks % 20 == 1:
                    self.get_logger().warn(f'EKF pose {dt:.2f} s old ({self.pose_fallbacks}x): frame skipped')
                return None
            return x + v * math.cos(yaw + w * dt / 2) * dt, y + v * math.sin(yaw + w * dt / 2) * dt, yaw + w * dt
        for a, b in zip(reversed(list(self.odom)[:-1]), reversed(self.odom)):
            if a[0] <= t <= b[0]:
                f = (t - a[0]) / max(b[0] - a[0], 1e-9)
                dyaw = math.atan2(math.sin(b[3] - a[3]), math.cos(b[3] - a[3]))
                return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), a[3] + f * dyaw
        return None

    def cluster(self, p):
        """8-connected clusters on a 2D grid. Returns (labels per point, per-cluster stats)."""
        ij = np.floor(p[:, :2] / self.cell).astype(np.int32)
        lo = ij.min(axis=0)
        ij -= lo
        grid = np.zeros(ij.max(axis=0) + 1, dtype=bool)
        grid[ij[:, 0], ij[:, 1]] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
        point_lab = lab[ij[:, 0], ij[:, 1]]
        dets = []
        if n == 0:
            return point_lab, dets
        idx = np.arange(1, n + 1)
        cnt = ndimage.sum(np.ones(len(p)), point_lab, idx)
        for k, c in zip(idx, cnt):
            if c < self.min_pts:
                continue
            q = p[point_lab == k]
            mn, mx = q[:, :2].min(axis=0), q[:, :2].max(axis=0)
            ext = mx - mn + self.cell
            top = float(q[:, 2].max() - self.ground_z)
            if max(ext) > self.max_len or top < self.min_h:
                continue                       # walls/barriers (static structure) or flat clutter
            dets.append({'label': int(k), 'c': (mn + mx) / 2, 'size': (ext[0], ext[1], top), 'pts': q[:, :2]})
        return point_lab, dets

    def hist_index(self, w):
        ij = np.floor((w - self.hist_origin + self.hist_half) / self.hist_res).astype(np.int64)
        ok = (ij >= 0).all(axis=1) & (ij < self.first_seen.shape[0]).all(axis=1)
        return ij, ok

    def old_fraction(self, wpts, t):
        """Fraction of an object's cells continuously occupied for at least history_age seconds."""
        ij, ok = self.hist_index(wpts)
        if not ok.any():
            return 1.0
        cells = np.unique(ij[ok], axis=0)
        r, c = cells[:, 0], cells[:, 1]
        old = (self.first_seen[r, c] < t - self.history_age) & (t - self.last_seen[r, c] <= self.history_gap)
        return float(np.mean(old))

    def mark_history(self, wpts, t):
        ij, ok = self.hist_index(wpts)
        ij = np.unique(ij[ok], axis=0)
        r, c = ij[:, 0], ij[:, 1]
        stale = t - self.last_seen[r, c] > self.history_gap
        self.first_seen[r[stale], c[stale]] = t
        self.last_seen[r, c] = t

    # -------------------------------------------------------------- update
    def on_cloud(self, msg):
        t_start = time.perf_counter()
        self._process(msg)
        dt = (time.perf_counter() - t_start) * 1000
        self.timing.append(dt)
        if len(self.timing) >= 100:
            self.get_logger().info(f'tracker: {np.mean(self.timing):.0f} ms/frame mean, '
                                   f'{np.max(self.timing):.0f} ms max, {len(self.tracks)} tracks')
            self.timing = []

    def _process(self, msg):
        t_msg = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t_msg < self.last_t - 1.0:       # simulation restarted: drop tracks and history
            self.get_logger().warn('clock jumped back: resetting tracks and occupancy history')
            self.tracks = []
            self.first_seen[:] = np.inf
            self.last_seen[:] = -np.inf
            self.hist_origin = None
            self.odom.clear()
        self.last_t = t_msg
        pts = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z')).astype(np.float32)
        pose = self.ego_pose(msg.header.stamp)
        if pose is None or len(pts) == 0:
            self.publish_static(msg.header, pts)
            return
        ex, ey, eyaw = pose
        c, s = math.cos(eyaw), math.sin(eyaw)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        near = (np.hypot(pts[:, 0], pts[:, 1]) < self.max_range) & (pts[:, 2] < self.ground_z + 3.0)
        cand = pts[near]
        dets, point_lab = [], None
        if len(cand):
            point_lab, dets = self.cluster(cand)
        rot = np.array([[c, -s], [s, c]])
        if self.hist_origin is None:
            self.hist_origin = np.array([ex, ey])
        for d in dets:   # detections into the odom frame
            d['w'] = rot @ d['c'] + (ex, ey)
            d['wpts'] = d['pts'] @ rot.T + (ex, ey)
            d['old'] = self.old_fraction(d['wpts'], t)
            d['range'] = float(np.hypot(*d['c']))
        if len(cand):
            self.mark_history(cand[:, :2] @ rot.T + (ex, ey), t)

        # predict, associate (greedy by distance), update
        dts = {id(tr): tr.predict(t) for tr in self.tracks}
        pairs = sorted(((np.linalg.norm(tr.x[:2] - d['w']), i, j)
                        for i, tr in enumerate(self.tracks) for j, d in enumerate(dets)))
        used_t, used_d = set(), set()
        for dist, i, j in pairs:
            tr = self.tracks[i]
            if i in used_t or j in used_d or dist > self.gate + tr.speed * 0.15:
                continue
            v_meas = None
            if tr.pts is not None and len(tr.pts) >= self.min_reg_pts and len(dets[j]['wpts']) >= self.min_reg_pts:
                dt = dts[id(tr)]
                shift = register(tr.pts, dets[j]['wpts'], tr.x[2:] * dt)
                if shift is not None and dt > 0:
                    v_meas = shift / dt
                    tr.reg_disp += shift
            tr.update(dets[j]['w'], dets[j]['size'], v_meas)
            tr.pts = dets[j]['wpts']
            tr.old, tr.range = dets[j]['old'], dets[j]['range']
            dets[j]['track'] = tr
            used_t.add(i)
            used_d.add(j)
        dt_frame = 0.1
        for i, tr in enumerate(self.tracks):
            if i not in used_t:
                tr.missed_time += dt_frame
        for j, d in enumerate(dets):
            if j not in used_d:
                tr = KalmanTrack(d['w'], d['size'], t)
                tr.pts = d['wpts']
                tr.old, tr.range = d['old'], d['range']
                self.tracks.append(tr)
                d['track'] = tr
        self.tracks = [tr for tr in self.tracks if tr.missed_time <= self.max_missed]

        for tr in self.tracks:            # hysteresis on the moving label
            if self.moving_criteria(tr):
                tr.votes = min(tr.votes + 1, self.moving_frames)
            else:
                tr.votes = max(tr.votes - 1, 0) if tr.moving else 0
            tr.moving = tr.votes >= self.moving_frames if not tr.moving else tr.votes > 0
        # Remove from the static world every detection that is, or overlaps, a moving object. A
        # fast car can miss its own track for a frame and spawn a new (not yet 'moving') one;
        # matching against the moving tracks' predicted boxes keeps it out of the map anyway.
        moving_labels = {d['label'] for d in dets if d['track'].moving}
        movers = [tr for tr in self.tracks if tr.moving]
        for d in dets:
            for tr in movers:
                hx, hy = tr.size[0] / 2 + 1.0, tr.size[1] / 2 + 1.0
                if abs(d['w'][0] - tr.x[0]) < hx and abs(d['w'][1] - tr.x[1]) < hy:
                    moving_labels.add(d['label'])
                    break
        self.publish_tracks(msg.header, ex, ey, c, s)
        static_pts = pts
        if moving_labels:
            keep = np.ones(len(pts), bool)
            keep[np.where(near)[0][np.isin(point_lab, list(moving_labels))]] = False
            static_pts = pts[keep]
        self.publish_static(msg.header, static_pts)

    def moving_criteria(self, tr):
        if tr.hits < self.confirm_hits or tr.speed < self.moving_speed:
            return False
        return (float(np.linalg.norm(tr.reg_disp)) > self.min_disp and tr.range < self.moving_range
                and tr.old < self.max_old_frac)

    # ------------------------------------------------------------- outputs
    def publish_tracks(self, header, ex, ey, c, s):
        arr = TrackArray()
        arr.header = header                       # base_link
        markers = MarkerArray()
        markers.markers.append(Marker(header=header, action=Marker.DELETEALL))
        for tr in self.tracks:
            if tr.hits < self.confirm_hits:
                continue
            dx, dy = tr.x[0] - ex, tr.x[1] - ey   # odom -> base_link
            bx, by = c * dx + s * dy, -s * dx + c * dy
            vx, vy = c * tr.x[2] + s * tr.x[3], -s * tr.x[2] + c * tr.x[3]
            m = Track()
            m.id, m.age, m.hits = tr.id, min(tr.age, 65535), min(tr.hits, 65535)
            m.position.x, m.position.y, m.position.z = float(bx), float(by), float(tr.size[2])
            m.velocity.x, m.velocity.y = float(vx), float(vy)
            m.size.x, m.size.y, m.size.z = (float(v) for v in tr.size)
            m.speed = tr.speed
            m.moving = tr.moving
            arr.tracks.append(m)
            color = (1.0, 0.2, 0.2) if m.moving else (0.6, 0.6, 0.6)
            box = Marker(header=header, ns='objects', id=tr.id, type=Marker.CUBE, action=Marker.ADD)
            box.pose.position.x, box.pose.position.y = float(bx), float(by)
            box.pose.position.z = self.ground_z + tr.size[2] / 2
            box.pose.orientation.w = 1.0
            box.scale.x, box.scale.y, box.scale.z = (max(float(v), 0.2) for v in tr.size)
            box.color.r, box.color.g, box.color.b, box.color.a = (*color, 0.5)
            markers.markers.append(box)
            if m.moving:
                arrow = Marker(header=header, ns='velocity', id=tr.id, type=Marker.ARROW, action=Marker.ADD)
                z = self.ground_z + tr.size[2] + 0.3
                arrow.points = [Point(x=float(bx), y=float(by), z=z), Point(x=float(bx + vx), y=float(by + vy), z=z)]
                arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.2, 0.4, 0.4
                arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = 1.0, 0.9, 0.1, 1.0
                markers.markers.append(arrow)
                txt = Marker(header=header, ns='label', id=tr.id, type=Marker.TEXT_VIEW_FACING, action=Marker.ADD)
                txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = float(bx), float(by), z + 0.6
                txt.scale.z = 0.8
                txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0
                txt.text = f'#{tr.id} {tr.speed:.1f} m/s'
                markers.markers.append(txt)
        self.pub_tracks.publish(arr)
        self.pub_markers.publish(markers)

    def publish_static(self, header, pts):
        self.pub_static.publish(point_cloud2.create_cloud_xyz32(header, pts))
        band = pts[(pts[:, 2] > self.ground_z + self.scan_min_h) & (pts[:, 2] < self.ground_z + self.scan_max_h)]
        scan = LaserScan()
        scan.header = header
        scan.angle_min = -math.pi
        scan.angle_increment = 2 * math.pi / self.bins
        scan.angle_max = scan.angle_min + scan.angle_increment * (self.bins - 1)
        scan.scan_time = 0.1
        scan.range_min, scan.range_max = 0.3, 60.0
        ranges = np.full(self.bins, np.inf, dtype=np.float32)
        if len(band):
            r = np.hypot(band[:, 0], band[:, 1])
            idx = ((np.arctan2(band[:, 1], band[:, 0]) + math.pi) / scan.angle_increment).astype(int) % self.bins
            k = (r > scan.range_min) & (r < scan.range_max)
            np.minimum.at(ranges, idx[k], r[k])
        scan.ranges = ranges.tolist()
        self.pub_scan.publish(scan)


def main():
    rclpy.init()
    node = ObjectTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
