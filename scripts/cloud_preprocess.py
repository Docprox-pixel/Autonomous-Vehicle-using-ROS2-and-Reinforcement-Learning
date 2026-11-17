#!/usr/bin/env python3
"""3D lidar pre-processing (simulator-agnostic: uses only the sensor contract).

Input : /lidar/points            PointCloud2, base_link, all lidars merged
Output: /perception/obstacle_points   PointCloud2, base_link: everything that is not road or car
        /perception/ground_points     PointCloud2, base_link: road surface (drivable)
        /scan_flat                    LaserScan,  base_link: obstacles between min_h and max_h
                                      above the road, flattened to 2D (input for SLAM)

Why /scan_flat: a single horizontal lidar layer only sees what happens to be at that height
(buildings at 2.3 m). Flattening every obstacle point between knee and roof height gives SLAM
and the costmap barriers, kerbs, parked cars, cones and poles as well.

Ground model: a road plane fitted in every scan (see fit_ground_plane), seeded with ground_z
from config/sensor_layout.yaml, so pitch, roll and gentle slopes do not turn road into obstacles.
"""
import math
import os

import numpy as np
from scipy.spatial import cKDTree

from auto_car_core.ground import height_above_road
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2
from sensor_msgs_py import point_cloud2


class CloudPreprocess(Node):
    def __init__(self):
        super().__init__('cloud_preprocess')
        layout_file = self.declare_parameter('sensor_layout', os.path.join(
            get_package_share_directory('auto_car'), 'config', 'sensor_layout.yaml')).value
        layout = yaml.safe_load(open(layout_file))
        self.ground_z = float(layout['ground_z'])
        self.box = layout['car_box']
        self.self_boxes = layout.get('self_boxes', [])   # mirrors etc. outside the body box
        self.ground_tol = self.declare_parameter('ground_tolerance', 0.20).value  # m above road = ground (drive-over height)
        self.min_h = self.declare_parameter('scan_min_height', 0.20).value          # m above road
        self.max_h = self.declare_parameter('scan_max_height', 2.52).value
        self.max_obstacle_h = self.declare_parameter('max_obstacle_height', 6.0).value
        self.bins = self.declare_parameter('scan_bins', 720).value
        self.scan_range = self.declare_parameter('scan_range_max', 60.0).value

        # Reliable publishers: they serve both reliable (octomap_server, slam_toolbox) and
        # best-effort subscribers; a best-effort publisher would silently never reach the former.
        self.pub_obs = self.create_publisher(PointCloud2, '/perception/obstacle_points', 5)
        self.pub_ground = self.create_publisher(PointCloud2, '/perception/ground_points', 5)
        self.pub_kerb = self.create_publisher(PointCloud2, '/perception/kerb_points', 5)
        self.pub_scan = self.create_publisher(LaserScan, '/scan_flat', 5)
        self.create_subscription(PointCloud2, '/lidar/points', self.on_cloud, qos_profile_sensor_data)
        self.get_logger().info(f'cloud_preprocess: ground_z={self.ground_z} m, scan band '
                               f'{self.min_h}-{self.max_h} m above road, {self.bins} bins')

    def fit_ground_plane(self, p):
        """Fit z = a*x + b*y + c to the road in every scan.

        A fixed ground height fails as soon as the car pitches: 0.5 deg of pitch (braking,
        suspension) moves the road 0.26 m at 30 m, and whole rings of asphalt become
        'obstacles'. Fitting a plane per scan absorbs pitch, roll and gentle slopes.
        Iteratively re-weighted least squares on candidate road points, so kerbs, cars and
        walls (which sit above the plane) are rejected after the first pass.
        """
        r = np.hypot(p[:, 0], p[:, 1])
        cand = p[(r > 2.0) & (r < 45.0) & (np.abs(p[:, 2] - self.ground_z) < 0.8)]
        if len(cand) < 100:
            return np.array([0.0, 0.0, self.ground_z])
        coef = np.array([0.0, 0.0, self.ground_z])
        for tol in (0.5, 0.25, 0.12):
            resid = cand[:, 2] - (coef[0] * cand[:, 0] + coef[1] * cand[:, 1] + coef[2])
            inl = cand[np.abs(resid) < tol]
            if len(inl) < 50:
                break
            A = np.c_[inl[:, 0], inl[:, 1], np.ones(len(inl))]
            coef = np.linalg.lstsq(A, inl[:, 2], rcond=None)[0]
        # Sanity: a car on a road never sees more than ~10 deg of slope or a 0.4 m height jump.
        if np.hypot(coef[0], coef[1]) > 0.18 or abs(coef[2] - self.ground_z) > 0.4:
            return np.array([0.0, 0.0, self.ground_z])
        return coef

    def height_above_local_ground(self, p, h_plane, cell=2.0, reach=6.0):
        """Height of every point above the road surface UNDER it (not one plane for the whole scan).
        Road height per 2 m cell = lowest return in that cell, if it lies near the fitted plane
        (within 0.6 m: road, not a wall); cells without road returns take the nearest road cell
        within `reach` m, else the plane."""
        ij = np.floor(p[:, :2] / cell).astype(np.int64)
        key = ij[:, 0] * 100003 + ij[:, 1]
        order = np.lexsort((p[:, 2], key))
        k_sorted = key[order]
        first = np.r_[True, k_sorted[1:] != k_sorted[:-1]]
        lowest = order[first]                                 # index of the lowest point per cell
        ok = np.abs(h_plane[lowest]) < 0.6
        if ok.sum() < 3:
            return h_plane
        g_idx = lowest[ok]
        centres = (ij[g_idx] + 0.5) * cell
        g_z = p[g_idx, 2]
        dist, nn = cKDTree(centres).query(p[:, :2])
        local = g_z[nn]
        plane_z = p[:, 2] - h_plane
        ground_z_here = np.where(dist < reach, local, plane_z)
        return p[:, 2] - ground_z_here

    @staticmethod
    def isolated(p, mask, radius=0.35, near=14.0):
        """Obstacle points with NO other obstacle point within `radius` (x-y) inside `near` metres.

        Anything solid that close returns many points (several per layer, several layers stacked,
        often seen by two lidars). A lone return is sensor noise: a single phantom point 1.1 m ahead
        of the bumper (nothing there on camera) froze the car for 90 s in the night drive test.
        Farther away a small object can legitimately be one point; there it is kept."""
        out = np.zeros(len(p), bool)
        idx = np.flatnonzero(mask & (np.hypot(p[:, 0], p[:, 1]) < near))
        if len(idx) == 0:
            return out
        q = p[mask][:, :2]
        n = cKDTree(q).query_ball_point(p[idx, :2], radius, return_length=True)
        out[idx[n < 2]] = True                   # the point itself counts once
        return out

    def on_cloud(self, msg):
        p = point_cloud2.read_points_numpy(msg, field_names=('x', 'y', 'z')).astype(np.float32)
        if len(p) == 0:
            return
        b = self.box
        self_hit = ((p[:, 0] > b['x'][0]) & (p[:, 0] < b['x'][1]) & (p[:, 1] > b['y'][0]) &
                    (p[:, 1] < b['y'][1]) & (p[:, 2] < b['z'][1] + 0.1))
        for sb in self.self_boxes:
            self_hit |= ((p[:, 0] > sb['x'][0]) & (p[:, 0] < sb['x'][1]) & (p[:, 1] > sb['y'][0]) &
                         (p[:, 1] < sb['y'][1]) & (p[:, 2] > sb['z'][0]) & (p[:, 2] < sb['z'][1]))
        p = p[~self_hit]

        a, b_, c = self.fit_ground_plane(p)
        h_plane = p[:, 2] - (a * p[:, 0] + b_ * p[:, 1] + c)  # height above the fitted road plane
        # ... corrected for slope changes: road surface grown cell by cell from the car (ramps,
        # bridges, crests); the old 'near the plane' cells made a bridge ramp an obstacle
        h = height_above_road(p, h_plane)
        ground = h < self.ground_tol
        obstacle = ~ground & (h < self.max_obstacle_h)
        obstacle &= ~self.isolated(p, obstacle)
        self.pub_ground.publish(point_cloud2.create_cloud_xyz32(msg.header, p[ground]))
        # KERBS: 6-20 cm above the local road (below the drive-over height, so not obstacles), within
        # 20 m: published as road-edge points (planner boundary). Works in tunnels and at night,
        # where the camera's colour rules fail. Isolated returns are dropped like obstacles.
        kerb = (h > 0.06) & (h < self.ground_tol) & (np.hypot(p[:, 0], p[:, 1]) < 20.0)
        kerb &= ~self.isolated(p, kerb, radius=0.4)
        kq = p[kerb].copy()
        kq[:, 2] = 0.5 + self.ground_z + 0.32           # flag height inside the planner band (virtual wall)
        self.pub_kerb.publish(point_cloud2.create_cloud_xyz32(msg.header, kq))
        # Obstacle z = height above the LOCAL road + the car's road level (ground_z): planner, shield
        # and passability then measure 'can I pass under / drive over' against the road at that
        # spot. With raw z, a tunnel ceiling seen from a downhill approach looked 0.5 m too low and
        # the car stopped at the tunnel entrance ('too low' for its 2.37 m roof lidar).
        q = p[obstacle].copy()
        q[:, 2] = h[obstacle] + self.ground_z
        self.pub_obs.publish(point_cloud2.create_cloud_xyz32(msg.header, q))

        band = p[(h > self.min_h) & (h < self.max_h)]
        scan = LaserScan()
        scan.header = msg.header                              # base_link
        scan.angle_min = -math.pi
        scan.angle_increment = 2 * math.pi / self.bins
        scan.angle_max = scan.angle_min + scan.angle_increment * (self.bins - 1)
        scan.scan_time = 0.1
        scan.range_min = 0.3
        scan.range_max = float(self.scan_range)
        ranges = np.full(self.bins, np.inf, dtype=np.float32)
        if len(band):
            r = np.hypot(band[:, 0], band[:, 1])
            idx = ((np.arctan2(band[:, 1], band[:, 0]) + math.pi) / scan.angle_increment).astype(int) % self.bins
            keep = (r > scan.range_min) & (r < scan.range_max)
            np.minimum.at(ranges, idx[keep], r[keep])
        scan.ranges = ranges.tolist()
        self.pub_scan.publish(scan)


def main():
    rclpy.init()
    node = CloudPreprocess()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
