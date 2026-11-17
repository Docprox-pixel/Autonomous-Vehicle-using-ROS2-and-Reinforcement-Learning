"""Live local map of the surroundings (rolling occupancy grid in the odom frame).

The lidar is used to BUILD A MAP of what is around the car, not to steer it directly: every scan
adds its obstacle points to a 0.25 m grid fixed in the odom frame; a cell counts as occupied only
after it has been seen in `confirm` scans (a one-scan ripple - ground return on a slope, rain, a
reflection - never reaches the planner), and cells inside the sensor range that are no longer seen
fade out (cars that drove away disappear). The planner plans along the route against this map.
(Standard local costmap idea: ROS navigation costmap_2d / Autoware occupancy grid, hit/miss counts.)
"""
import math

import numpy as np

_OFF = 1 << 20


class LiveMap:
    def __init__(self, cell=0.25, confirm=2, cap=5, rng=30.0):
        self.cell, self.confirm, self.cap, self.rng = cell, confirm, cap, rng
        self.keys = np.zeros(0, np.int64)       # sorted cell keys
        self.hits = np.zeros(0, np.int16)
        self.z = np.zeros(0, np.float32)

    def _key(self, ix, iy):
        return (ix.astype(np.int64) + _OFF) * (2 * _OFF) + (iy.astype(np.int64) + _OFF)

    def _xy(self, keys):
        ix = keys // (2 * _OFF) - _OFF
        iy = keys % (2 * _OFF) - _OFF
        return (ix + 0.5) * self.cell, (iy + 0.5) * self.cell

    def update(self, pts_base, pose):
        """Add one scan (Nx3 base_link points) taken at odom pose (x, y, yaw)."""
        x, y, yaw = pose
        c, s = math.cos(yaw), math.sin(yaw)
        if pts_base is not None and len(pts_base):
            p = pts_base[np.hypot(pts_base[:, 0], pts_base[:, 1]) < self.rng]
        else:
            p = np.zeros((0, 3), np.float32)
        wx = x + c * p[:, 0] - s * p[:, 1]
        wy = y + s * p[:, 0] + c * p[:, 1]
        k = self._key(np.floor(wx / self.cell), np.floor(wy / self.cell))
        k_new, inv = np.unique(k, return_inverse=True)
        z_new = np.full(len(k_new), -np.inf, np.float32)
        np.maximum.at(z_new, inv, p[:, 2].astype(np.float32))
        # misses: known cells inside the sensor range that this scan did not hit fade by one
        if len(self.keys):
            cx, cy = self._xy(self.keys)
            inside = np.hypot(cx - x, cy - y) < self.rng - 2.0
            hit_now = np.isin(self.keys, k_new, assume_unique=True)
            self.hits = self.hits - (inside & ~hit_now).astype(np.int16)
        # hits: existing cells +1, new cells start at 1
        pos = np.searchsorted(self.keys, k_new)
        exists = (pos < len(self.keys)) & (self.keys[np.minimum(pos, len(self.keys) - 1)] == k_new) \
            if len(self.keys) else np.zeros(len(k_new), bool)
        if exists.any():
            ip = pos[exists]
            self.hits[ip] = np.minimum(self.hits[ip] + 1, self.cap)
            self.z[ip] = z_new[exists]
        add = ~exists
        keys = np.r_[self.keys, k_new[add]]
        hits = np.r_[self.hits, np.ones(int(add.sum()), np.int16)]
        z = np.r_[self.z, z_new[add]]
        # drop faded cells and cells far behind
        cx, cy = self._xy(keys)
        keep = (hits > 0) & (np.hypot(cx - x, cy - y) < self.rng + 10.0)
        o = np.argsort(keys[keep])
        self.keys, self.hits, self.z = keys[keep][o], hits[keep][o], z[keep][o]

    def occupied(self, pose):
        """Confirmed obstacle cells as Nx3 base_link points (cell centres, highest z seen)."""
        m = self.hits >= self.confirm
        if not m.any():
            return np.zeros((0, 3), np.float32)
        x, y, yaw = pose
        cx, cy = self._xy(self.keys[m])
        dx, dy = cx - x, cy - y
        c, s = math.cos(yaw), math.sin(yaw)
        return np.c_[c * dx + s * dy, -s * dx + c * dy, self.z[m]].astype(np.float32)
