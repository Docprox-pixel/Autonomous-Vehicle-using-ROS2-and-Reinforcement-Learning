#!/usr/bin/env python3
"""Road perception from the front camera: where is road, where is the centre line.

The lidar only sees obstacles. Grass, gravel and the oncoming lanes are flat and look exactly
as free as our own lane, so a planner that knows only obstacles cuts corners into the other lane
or leaves the road where there is no barrier. This node gives the planner that missing knowledge.

Input : /camera/image_raw (bgra8), /camera/camera_info, /odometry/filtered
Output: /perception/road_boundary  PointCloud2 (base_link): VIRTUAL walls for the planner only:
                                   off-road ground + the yellow centre line (do not cross into
                                   oncoming traffic). The safety shield never uses these.
        /perception/road_bev       mono8 bird's-eye debug image (0 unknown, 100 road,
                                   200 off-road, 255 centre line), 0.5 m cells, car at the bottom

Pipeline (per frame, image downsampled 4x):
  1. classify pixels: asphalt (dark, unsaturated) / lane markings (white, yellow) = road;
     everything else below the horizon = off-road. Simple colour rules tuned for the Webots
     city; a learned segmentation model can replace classify() with the same outputs.
  2. inverse perspective mapping: camera height and focal length put each pixel on the road
     plane (x ahead, y left) in base_link.
  3. yellow pixels form connected blobs in the bird's-eye view; a blob elongated ALONG the
     driving direction is the centre line (a wall), one across it is a crosswalk (drivable).
  4. memory: labels are kept in a world (odom) grid for `memory` seconds, so the car still
     knows the road beside it while turning, where the camera does not look.
"""
import math
import os
from collections import deque

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from nav_msgs.msg import Odometry
from rclpy.node import Node
from scipy import ndimage
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from sensor_msgs_py import point_cloud2

UNKNOWN, ROAD, OFFROAD, CENTRE = 0, 1, 2, 3


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def to_hsv(bgr):
    """Vectorised BGR -> HSV (OpenCV ranges: H 0-180, S/V 0-255), avoids a cv2 dependency."""
    b, g, r = (bgr[..., i].astype(np.float32) for i in range(3))
    mx, mn = np.maximum(np.maximum(r, g), b), np.minimum(np.minimum(r, g), b)
    d = mx - mn + 1e-6
    h = np.where(mx == r, (g - b) / d % 6, np.where(mx == g, (b - r) / d + 2, (r - g) / d + 4)) * 30
    s = np.where(mx > 0, d / (mx + 1e-6) * 255, 0)
    return h, s, mx


def classify(bgr):
    """Return (road, yellow) boolean masks. Everything else is off-road."""
    h, s, v = to_hsv(bgr)
    # Asphalt: dark and grey. In deep shadow it is nearly black and picks up the sky's blue tint,
    # so saturation is meaningless there: anything darker than V 40 on the ground is road
    # (shadowed sand/grass stays at V ~90+). Found when shadowed road looked like a wall.
    asphalt = (v < 40) | ((s < 34) & (v < 80))
    # Lane markings: unsaturated and bright. Saturation separates them from sand (sunlit sand S ~39,
    # shadowed sand V ~97). A brightness-only rule (V > 180) missed slightly dim dashes, turning the
    # dashed lane line into a row of little walls in the middle of the road.
    white = (s < 25) & (v > 120)
    yellow = (h > 12) & (h < 38) & (s > 90) & (v > 110)
    return asphalt | white | yellow, yellow


def centre_line(x, y, max_range=30.0):
    """Yellow points (base_link) that belong to centre lines, in ANY orientation.

    Blobs on a 0.4 m grid WITHOUT dilation: crosswalk stripes stay separate (gaps between them)
    and are short (~1.9 m measured); the centre line is one long thin blob (18 m x 1.0 m
    measured). Rule: length > 4.5 m and width < 1.5 m.
    Orientation is not used: at a junction the CROSS street's centre line runs across our path and
    must be a wall too, otherwise a turn swings wide into the oncoming lanes (seen in testing).
    """
    if len(x) < 10:
        return np.zeros((0, 2))
    g = 0.4
    ix, iy = np.floor(x / g).astype(int), np.floor((y + 30) / g).astype(int)
    grid = np.zeros((int(max_range / g) + 20, int(60 / g) + 2), bool)
    ok = (ix >= 0) & (ix < grid.shape[0]) & (iy >= 0) & (iy < grid.shape[1])
    grid[ix[ok], iy[ok]] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
    keep = []
    for k in range(1, n + 1):
        cells = np.argwhere((lab == k) & grid)
        if len(cells) < 6:
            continue
        w, _ = np.linalg.eigh(np.cov((cells * g).T))
        length, width = 4 * math.sqrt(max(w[1], 0)), 4 * math.sqrt(max(w[0], 0))
        if length > 4.5 and width < 1.5:
            keep.append(np.c_[cells[:, 0] * g + g / 2, cells[:, 1] * g + g / 2 - 30])
    return np.concatenate(keep) if keep else np.zeros((0, 2))


class RoadPerception(Node):
    def __init__(self):
        super().__init__('road_perception')
        layout = yaml.safe_load(open(self.declare_parameter('sensor_layout', os.path.join(
            get_package_share_directory('auto_car'), 'config', 'sensor_layout.yaml')).value))
        roof = next(l for l in layout['lidars'] if l['name'] == 'lidar')
        self.cam_x = roof['xyz'][0]                          # camera shares the roof mount
        self.cam_h = roof['xyz'][2] - layout['ground_z']     # height above the road
        p = self.declare_parameter
        self.down = p('downsample', 4).value
        self.max_range = p('max_range', 30.0).value
        self.cell = p('cell', 0.5).value
        self.memory = p('memory', 4.0).value
        self.publish_range = p('publish_range', 30.0).value
        self.hood_rows = p('hood_rows', 20).value             # bottom rows showing the bonnet
        self.f = None
        self.odom = deque(maxlen=300)
        half, n = 400.0, int(800.0 / self.cell)
        self.half = half
        self.label = np.zeros((n, n), np.uint8)
        self.seen = np.full((n, n), -np.inf, np.float32)
        self.origin = None
        self.last_t = -np.inf
        self.pub = self.create_publisher(PointCloud2, '/perception/road_boundary', 5)
        self.pub_bev = self.create_publisher(Image, '/perception/road_bev', 2)
        # Lane-line measurement for lane-level map matching: lateral distance (m, base_link y, left
        # +) from the car to the LEFT lane line (white dashed or yellow centre line), its angle, and
        # the number of supporting marking pixels.
        from std_msgs.msg import String as _S
        self.pub_lane = self.create_publisher(_S, '/perception/lane_marking', 5)
        self.create_subscription(CameraInfo, '/camera/camera_info', self.on_info, 2)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 50)
        self.create_subscription(Image, '/camera/image_raw', self.on_image, 2)
        self.get_logger().info(f'road_perception: camera {self.cam_h:.2f} m above road, range {self.max_range} m')

    def on_info(self, m):
        self.f = m.k[0]

    def on_odom(self, m):
        self.odom.append((m.header.stamp.sec + m.header.stamp.nanosec * 1e-9, m.pose.pose.position.x,
                          m.pose.pose.position.y, yaw_of(m.pose.pose.orientation)))

    def pose_at(self, t):
        if not self.odom:
            return None
        best = min(self.odom, key=lambda o: abs(o[0] - t))
        return best[1:] if abs(best[0] - t) < 0.15 else None

    def idx(self, wx, wy):
        i = np.floor((wx - self.origin[0] + self.half) / self.cell).astype(np.int64)
        j = np.floor((wy - self.origin[1] + self.half) / self.cell).astype(np.int64)
        ok = (i >= 0) & (j >= 0) & (i < self.label.shape[0]) & (j < self.label.shape[1])
        return i, j, ok

    def on_image(self, m):
        if self.f is None:
            return
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        if t < self.last_t - 1.0:
            # Simulation restarted (clock jumped back). Old cells would look 'recent' forever and
            # become ghost walls: this stopped the car on curves in two drive tests.
            self.get_logger().warn(f'clock jumped back {self.last_t:.1f} -> {t:.1f} s: clearing road memory')
            self.label[:] = UNKNOWN
            self.seen[:] = -np.inf
            self.origin = None
            self.odom.clear()
        self.last_t = t
        pose = self.pose_at(t)
        if pose is None:
            return
        if self.origin is None:
            self.origin = pose[:2]
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)[::self.down, ::self.down, :3]
        H, W = img.shape[:2]
        f, cx, cy = self.f / self.down, W / 2.0, H / 2.0
        rows = np.arange(H)
        min_row = cy + f * self.cam_h / self.max_range
        use = (rows > min_row) & (rows < H - self.hood_rows // self.down)
        # Fog: beyond the visibility range a row becomes one flat fog colour (measured: brightness
        # spread 0.1-1.0 at V ~94, versus >= 10 on real road). Such rows carry no information and
        # would read as 'off-road' -> a false wall across the road. Skip them. Night rows are dark
        # (V < 40), not flat-and-bright, so they are kept.
        v_rows = img.max(axis=2).astype(np.float32)
        flat = (v_rows.std(axis=1) < 2.0) & (np.median(v_rows, axis=1) > 40)
        use &= ~flat
        if not use.any():
            return
        road, yellow = classify(img[use])
        hh, ss, vvv = to_hsv(img[use])
        marking = ((ss < 25) & (vvv > 120)) | yellow             # white or yellow paint
        vv, uu = np.meshgrid(rows[use], np.arange(W), indexing='ij')
        X = f * self.cam_h / (vv - cy)                     # ground distance ahead of the camera
        bx, by = self.cam_x + X, -(uu - cx) * X / f        # base_link
        # bird's-eye cells for this frame (base_link), then blob analysis of yellow
        lab = np.where(road, ROAD, OFFROAD).astype(np.uint8)
        lab[yellow] = ROAD
        centre = self.centre_line(bx[yellow], by[yellow])
        self.measure_left_line(m.header, bx[marking], by[marking])
        x, yaw_ = pose[:2], pose[2]
        c, s = math.cos(yaw_), math.sin(yaw_)
        wx, wy = x[0] + c * bx - s * by, x[1] + s * bx + c * by
        i, j, ok = self.idx(wx.ravel(), wy.ravel())
        flat = lab.ravel()[ok]
        i, j = i[ok], j[ok]
        # a cell is off-road only if most of its pixels are off-road (edges stay drivable)
        key = i * self.label.shape[1] + j
        uniq, inv = np.unique(key, return_inverse=True)
        off = np.bincount(inv, weights=(flat == OFFROAD)) / np.bincount(inv)
        ci, cj = uniq // self.label.shape[1], uniq % self.label.shape[1]
        is_off = off > 0.6
        is_off &= ~self.small_patches(ci, cj, is_off)   # a real road edge is big; specks are noise
        self.label[ci, cj] = np.where(is_off, OFFROAD, ROAD)
        self.seen[ci, cj] = t
        if len(centre):
            ex, ey = x[0] + c * centre[:, 0] - s * centre[:, 1], x[1] + s * centre[:, 0] + c * centre[:, 1]
            ii, jj, okc = self.idx(ex, ey)
            self.label[ii[okc], jj[okc]] = CENTRE
            self.seen[ii[okc], jj[okc]] = t
        self.publish(m.header, t, pose)

    def centre_line(self, x, y):
        return centre_line(x, y, self.max_range)

    def measure_left_line(self, header, x, y):
        """RANSAC line through marking pixels 3-14 m ahead, 0.4-3.6 m to the LEFT, roughly along the
        driving direction (+/-15 deg): the left lane line (centre line of a two-way road). Dashes
        have gaps, RANSAC does not care. Published as JSON on /perception/lane_marking."""
        import json
        from std_msgs.msg import String as _S
        sel = (x > 3.0) & (x < 14.0) & (y > 0.4) & (y < 3.6)
        px, py = x[sel], y[sel]
        out = {'valid': False}
        if len(px) >= 25:
            rng = np.random.default_rng(0)
            best = (0, None)
            for _ in range(60):
                i, j = rng.integers(len(px), size=2)
                if abs(px[i] - px[j]) < 1.5:
                    continue
                a = (py[j] - py[i]) / (px[j] - px[i])
                if abs(a) > math.tan(math.radians(15)):
                    continue
                b = py[i] - a * px[i]
                inl = np.abs(py - (a * px + b)) < 0.15
                if inl.sum() > best[0]:
                    best = (int(inl.sum()), inl)
            if best[1] is not None and best[0] >= 20:
                inl = best[1]
                a, b = np.polyfit(px[inl], py[inl], 1)
                span = px[inl].max() - px[inl].min()
                if span > 3.0:
                    out = {'valid': True, 'y_at_body': float(a * 1.45 + b), 'angle': float(math.atan(a)),
                           'support': best[0], 'span': float(span)}
        self.pub_lane.publish(_S(data=json.dumps(out)))

    def small_patches(self, ci, cj, is_off, min_cells=6):
        """True for off-road cells (this frame) in connected patches smaller than min_cells
        (6 cells of 0.5 m = 1.5 m^2)."""
        out = np.zeros(len(ci), bool)
        if not is_off.any():
            return out
        i0, j0 = ci.min(), cj.min()
        grid = np.zeros((ci.max() - i0 + 1, cj.max() - j0 + 1), bool)
        grid[ci[is_off] - i0, cj[is_off] - j0] = True
        lab, n = ndimage.label(grid, structure=np.ones((3, 3)))
        if n == 0:
            return out
        sizes = np.bincount(lab.ravel())
        small = sizes < min_cells
        small[0] = False
        return is_off & small[lab[ci - i0, cj - j0]]

    def publish(self, header, t, pose):
        x, y, yaw_ = pose
        recent = (t - self.seen) < self.memory
        walls = recent & ((self.label == OFFROAD) | (self.label == CENTRE))
        ii, jj = np.nonzero(walls)
        wx = (ii + 0.5) * self.cell - self.half + self.origin[0]
        wy = (jj + 0.5) * self.cell - self.half + self.origin[1]
        c, s = math.cos(yaw_), math.sin(yaw_)
        dx, dy = wx - x, wy - y
        bx, by = c * dx + s * dy, -s * dx + c * dy
        near = np.hypot(bx, by) < self.publish_range
        pts = np.c_[bx[near], by[near], np.full(near.sum(), 0.5)].astype(np.float32)
        header.frame_id = 'base_link'
        self.pub.publish(point_cloud2.create_cloud_xyz32(header, pts))
        # debug bird's-eye image: 40 x 40 m around the car, car at bottom centre
        n = int(40 / self.cell)
        bev = np.zeros((n, n), np.uint8)
        sel = recent & (self.label > 0)
        ii, jj = np.nonzero(sel)
        wx = (ii + 0.5) * self.cell - self.half + self.origin[0]
        wy = (jj + 0.5) * self.cell - self.half + self.origin[1]
        dx, dy = wx - x, wy - y
        bx, by = c * dx + s * dy, -s * dx + c * dy
        r, q = (n - 1 - np.floor(bx / self.cell)).astype(int), (n // 2 - np.floor(by / self.cell)).astype(int)
        okb = (r >= 0) & (r < n) & (q >= 0) & (q < n) & (bx > -2)
        bev[r[okb], q[okb]] = np.array([0, 100, 200, 255], np.uint8)[self.label[ii[okb], jj[okb]]]
        img = Image(header=header, height=n, width=n, encoding='mono8', step=n, data=bev.tobytes())
        self.pub_bev.publish(img)


def main():
    rclpy.init()
    node = RoadPerception()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
