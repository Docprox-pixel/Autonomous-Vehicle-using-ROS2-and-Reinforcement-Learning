"""Vehicle geometry and safety maths shared by the safety shield and the local planner.

Pure Python + numpy (no ROS), so it is unit-tested directly (test/test_shield_logic.py) and is
simulator-agnostic. Frame: base_link (x forward, y left, origin = rear-axle centre).
"""
import math
from dataclasses import dataclass

import numpy as np

CAR_FRONT = 4.05       # base_link x of the front bumper (config/sensor_layout.yaml car_box)
CAR_REAR = -1.15
HALF_WIDTH = 1.0
# Height-aware body: above the body roof (base_link z 1.45 + 0.15 m) only the roof sensor mount
# sticks up, MOUNT_HALF_WIDTH to each side of the centre line. The curved tunnel ceiling over the
# sidewalk (2.17 m) was treated as a wall beside the car (2.37 m box) and stopped it in the tunnel.
ROOF_Z = 1.60
MOUNT_HALF_WIDTH = 0.6
# Between the window line (base_link z 1.0 = 1.32 m above the road) and the roof the SUV cabin is
# narrower than the body: about 0.85 m each side (a roadside sign plate at 1.5 m beside the cabin
# was treated as touching the 2.0 m wide body box).
CABIN_Z = 1.00
CABIN_HALF_WIDTH = 0.85


def compact(pts, cell=0.1):
    """One point per 10 cm (x, y) cell: the LOWEST one (the part a bumper/body meets first). Same
    geometry for the checks, ~4x fewer points; planner and shield both use it, so they agree."""
    if len(pts) < 2000:
        return pts
    p = pts[np.argsort(pts[:, 2], kind='stable')]
    _, idx = np.unique(np.floor(p[:, :2] / cell).astype(np.int32), axis=0, return_index=True)
    return p[idx]


def _narrow_above(pts_z, by, lat_margin, bx=None):
    """(poses, points) mask of points that CANNOT touch the car because they are above its body at
    that place (BmwX5 side profile, base_link z = road + 0.32 m, margin 0.15 m):
      hood   (x > 2.8 m)         top z 0.80  (~1.1 m above the road)
      cabin  (-0.9 .. 2.8 m)     top z 1.45, and above the window line (z 1.0) only 0.85 m wide
      rear   (x < -0.9 m)        top z 1.30
      roof sensor mount          z up to 2.05 (2.37 m above the road), |y| < 0.6, x -0.3 .. 1.8
    (A tunnel ceiling at 2.17 m in front of the bumper, and a sign plate at 1.5 m beside the cabin,
    were treated as touching a 2.37 m tall, 2.0 m wide box and stopped the car.)"""
    a = np.abs(by)
    z = pts_z[None, :] if np.ndim(by) == 2 else pts_z
    vm = 0.15
    if bx is None:
        hi = z > ROOF_Z
        cab = (z > CABIN_Z) & ~hi
        return (hi & (a > MOUNT_HALF_WIDTH + lat_margin)) | (cab & (a > CABIN_HALF_WIDTH + lat_margin))
    top = np.where(bx > 2.8, 0.80, np.where(bx < -0.9, 1.30, 1.45)) + vm
    mount = (a < MOUNT_HALF_WIDTH + lat_margin) & (bx > -0.3 - lat_margin) & (bx < 1.8 + lat_margin)
    above = (z > top) & ~mount
    beside_cabin = (z > CABIN_Z) & (a > CABIN_HALF_WIDTH + lat_margin) & (bx > -0.9) & (bx <= 2.8)
    return above | beside_cabin


@dataclass
class Params:
    a_brake: float = 3.2          # m/s^2: 85% of the measured 3.77 (test/braking_test.py, BmwX5)
    latency: float = 0.3          # s: measured 0.17 actuation + one perception frame
    margin: float = 2.0           # m kept to a static obstacle
    corridor_margin: float = 0.4  # m added around the car body (sides and rear) in the swept check
    horizon: float = 4.0          # s of dynamic prediction
    dt: float = 0.1
    ttc_stop: float = 1.5         # s
    ttc_slow: float = 3.5         # s
    a_lead_max: float = 8.0       # m/s^2 worst-case braking of a car in front (less credit)
    wheelbase: float = 2.995
    hard_margin: float = 0.15     # m around the body that may never be entered
    # Vertical envelope (config/sensor_layout.yaml 'vehicle'): only points in this band can touch
    # the car. Below: low enough to drive over (<= 0.15 m above the road; BmwX5 ground clearance
    # 0.21 m; raised pedestrian crossings are 0.15-0.17 m slabs, driven over). Above: high enough to pass
    # under (>= 2.52 m above the road = top of the roof lidar
    # 2.37 m + 0.15 m). The old band (0.25 .. 2.22 m) ignored a 0.22 m block the bumper would hit
    # and a 2.3 m barrier the roof lidar would hit.
    min_h: float = -0.12          # base_link z: road (-0.32) + 0.20 m (drive-over height)
    max_h: float = 2.20           # base_link z: road (-0.32) + 2.52 m


def path_poses(v, steer, p, t):
    """Ego pose (x, y, yaw) after t seconds along the bicycle-model arc from base_link."""
    s = v * t
    k = math.tan(steer) / p.wheelbase
    if abs(k) < 1e-4:
        return s, 0.0, 0.0
    yaw = s * k
    return math.sin(yaw) / k, (1 - math.cos(yaw)) / k, yaw


def static_distance(points, steer, p, return_point=False, max_travel=60.0, step=0.25, contact=True):
    """Distance the car can travel along its predicted arc before its BODY touches an obstacle.

    Swept footprint: the full car rectangle (+ margins) is moved along the bicycle-model arc
    and every obstacle point is tested against it at each pose. In a turn the front corner
    swings OUT and the rear corner cuts IN; a band around the rear-axle path misses both
    (a pole on the outside of a curve, a kerb or cyclist on the inside).
    Points at the start (s = 0):
      * inside the car body itself      -> ignored (self-hits / sensor noise)
      * within hard_margin of the body  -> contact NOW, distance 0 (the car must not move on);
        needs >= 2 points (a single stray return at the body edge froze the car at a stop line)
      * within the soft margin only     -> ignored for this check, so the car can still drive
        away from something it passes closely. (Earlier versions ignored the whole margin,
        which made obstacles closer than 0.4 m invisible: a scrape found in the drive test.)
    """
    none = (math.inf, None) if return_point else math.inf
    if points is None or len(points) == 0:
        return none
    pts = points[(points[:, 2] > p.min_h) & (points[:, 2] < p.max_h)]
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < max_travel + CAR_FRONT + 3.0]
    if len(pts) == 0:
        return none
    pts = compact(pts)
    if contact:
        hw = HALF_WIDTH + p.corridor_margin
        x0, x1 = CAR_REAR - p.corridor_margin, CAR_FRONT
    else:       # virtual walls (lane lines): the car's real body, no safety margin
        hw, x0, x1 = HALF_WIDTH + 0.05, CAR_REAR, CAR_FRONT

    s_arr = np.arange(0.0, max_travel + 1e-9, step)                 # rear-axle travel
    k = math.tan(steer) / p.wheelbase
    yaw = s_arr * k
    if abs(k) < 1e-4:
        ex, ey = s_arr, np.zeros_like(s_arr)
    else:
        ex, ey = np.sin(yaw) / k, (1 - np.cos(yaw)) / k
    c, sn = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    dx, dy = pts[None, :, 0] - ex[:, None], pts[None, :, 1] - ey[:, None]
    bx, by = c * dx + sn * dy, -sn * dx + c * dy                     # points in the future body frame
    inside = (bx > x0) & (bx < x1) & (np.abs(by) < hw)               # (poses, points)
    inside &= ~_narrow_above(pts[:, 2], by, hw - HALF_WIDTH, bx)    # body profile: hood/cabin/mount
    b0x, b0y = bx[0], by[0]                                          # points relative to the car now
    in_body = (b0x > CAR_REAR) & (b0x < CAR_FRONT) & (np.abs(b0y) < HALF_WIDTH)
    hm = p.hard_margin
    in_hard = ((b0x > CAR_REAR - hm) & (b0x < CAR_FRONT + hm) & (np.abs(b0y) < HALF_WIDTH + hm)) & ~in_body
    in_hard &= ~_narrow_above(pts[:, 2], b0y[None, :], p.hard_margin, b0x[None, :])[0]
    if contact and in_hard.sum() >= 2:     # a real obstacle this close returns several points; one stray return must not freeze the car
        return (0.0, pts[np.argmax(in_hard)]) if return_point else 0.0
    inside &= ~inside[0][None, :]                                    # soft-margin overlap at s = 0
    if not contact:
        # virtual line already at the car (it drifted onto it): ignore line points within 0.5 m of
        # the body now, so every arc that steers back into the lane stays possible
        near0 = (b0x > x0 - 0.5) & (b0x < x1 + 0.5) & (np.abs(b0y) < hw + 0.5)
        inside &= ~near0[None, :]
    hit = inside.any(axis=1)
    if not hit.any():
        return none
    i = int(np.argmax(hit))
    cand = pts[inside[i]]

    def overlaps(sv):
        """Does the footprint at travel sv overlap any candidate point?"""
        yv = sv * k
        if abs(k) < 1e-4:
            px, py = sv, 0.0
        else:
            px, py = math.sin(yv) / k, (1 - math.cos(yv)) / k
        cy, sy = math.cos(yv), math.sin(yv)
        qx, qy = cand[:, 0] - px, cand[:, 1] - py
        qbx, qby = cy * qx + sy * qy, -sy * qx + cy * qy
        return bool(((qbx > x0) & (qbx <= x1) & (np.abs(qby) < hw)).any())

    lo, hi = (float(s_arr[i - 1]) if i > 0 else 0.0), float(s_arr[i])   # clear .. touching
    for _ in range(6):                                                  # bisect to ~4 mm
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if overlaps(mid) else (mid, hi)
    d = hi
    if return_point:
        return d, cand[0]
    return d


def path_static_distance(points, path, p, max_travel=None):
    """Free travel along the PLANNED PATH (Nx2, base_link, starting near the car) with the same
    contact rules as static_distance: the footprint (plus corridor margin) is swept along the path
    the car will actually drive, not along a constant-curvature arc (in a curving tunnel the arc of
    the current steering runs into the outer wall although the path itself is clear).
    Returns None when the path does not start at the car (use the arc check then)."""
    if path is None or len(path) < 3:
        return None
    max_travel = 30.0 if max_travel is None else max_travel
    i0 = int(np.argmin(np.hypot(path[:, 0], path[:, 1])))
    P = np.array(path[i0:], float)
    P = P[P[:, 0] > -0.5] if (P[:, 0] > -0.5).sum() >= 3 else P   # no points behind the car
    if len(P) < 3 or np.hypot(*P[0]) > 2.0:
        return None
    # path heading (smoothed over ~1 m), before blending the positions onto the car
    s_raw = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    gx, gy = np.gradient(P[:, 0], s_raw, edge_order=1), np.gradient(P[:, 1], s_raw, edge_order=1)
    h_raw = np.unwrap(np.arctan2(gy, gx))
    # the car is not exactly on the path: it converges over ~10 m, so the swept poses start at the
    # car and blend onto the path (joining the car straight to the nearest path point swung the
    # footprint sideways into the tunnel wall)
    s0 = s_raw
    w = np.clip(1.0 - s0 / 6.0, 0.0, 1.0)[:, None]          # the car can turn onto the path within ~6 m
    P = P + w * (np.array([0.0, 0.0]) - P[0])
    h_blend = (1.0 - w[:, 0]) * h_raw                         # car heading 0 -> path heading over 6 m
    seg = np.hypot(*np.diff(P, axis=0).T)
    sP = np.r_[0.0, np.cumsum(seg)]
    if sP[-1] < 2.0:
        return None
    s_arr = np.arange(0.0, min(max_travel, sP[-1]) + 1e-9, 0.1)
    ex, ey = np.interp(s_arr, sP, P[:, 0]), np.interp(s_arr, sP, P[:, 1])
    hd = np.interp(s_arr, sP, h_blend)                             # starts at the car's own heading
    if points is None or len(points) == 0:
        return math.inf
    pts = points[(points[:, 2] > p.min_h) & (points[:, 2] < p.max_h)]
    pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < s_arr[-1] + CAR_FRONT + 3.0]
    if len(pts) == 0:
        return math.inf
    pts = compact(pts)
    hw = HALF_WIDTH + p.corridor_margin
    x0, x1 = CAR_REAR - p.corridor_margin, CAR_FRONT
    c, sn = np.cos(hd)[:, None], np.sin(hd)[:, None]
    dx, dy = pts[None, :, 0] - ex[:, None], pts[None, :, 1] - ey[:, None]
    bx, by = c * dx + sn * dy, -sn * dx + c * dy
    inside = (bx > x0) & (bx < x1) & (np.abs(by) < hw)
    inside &= ~_narrow_above(pts[:, 2], by, hw - HALF_WIDTH, bx)
    b0x, b0y = bx[0], by[0]
    in_body = (b0x > CAR_REAR) & (b0x < CAR_FRONT) & (np.abs(b0y) < HALF_WIDTH)
    hm = p.hard_margin
    in_hard = ((b0x > CAR_REAR - hm) & (b0x < CAR_FRONT + hm) & (np.abs(b0y) < HALF_WIDTH + hm)) & ~in_body
    in_hard &= ~_narrow_above(pts[:, 2], b0y[None, :], hm, b0x[None, :])[0]
    if in_hard.sum() >= 2:
        return 0.0
    inside &= ~inside[0][None, :]
    hit = inside.any(axis=1)
    return float(s_arr[int(np.argmax(hit))]) if hit.any() else math.inf


def time_to_collision(v, steer, tracks, p):
    """First time (s) a moving track overlaps the ego footprint, simulating both forward."""
    best = math.inf
    for tr in tracks:
        hx, hy = tr['size'][0] / 2 + 0.3, tr['size'][1] / 2 + 0.3
        for i in range(1, int(p.horizon / p.dt) + 1):
            t = i * p.dt
            ex, ey, eyaw = path_poses(v, steer, p, t)
            ox, oy = tr['pos'][0] + tr['vel'][0] * t, tr['pos'][1] + tr['vel'][1] * t
            # object centre in the ego's future body frame
            c, s = math.cos(eyaw), math.sin(eyaw)
            bx, by = c * (ox - ex) + s * (oy - ey), -s * (ox - ex) + c * (oy - ey)
            r = max(hx, hy)            # orientation unknown: use the bounding circle radius
            if CAR_REAR - r < bx < CAR_FRONT + r and abs(by) < HALF_WIDTH + r:
                best = min(best, t)
                break
    return best


def safe_speed(d, p):
    """Largest v with  v*latency + v^2/(2a) <= d - margin."""
    room = d - p.margin
    if room <= 0:
        return 0.0
    a, tl = p.a_brake, p.latency
    return a * (-tl + math.sqrt(tl * tl + 2 * room / a))


def evaluate(cmd_v, steer, ego_v, points, moving_tracks, p=Params(), path=None):
    """Return (safe_speed, info). Pure function: no ROS, no state.
    path: the planner's planned path (base_link); when given (and it starts at the car) the static
    check sweeps the footprint along it instead of along the constant arc of the current steering."""
    v_plan = max(cmd_v, ego_v, 0.0)
    d, nearest = static_distance(points, steer, p, return_point=True)
    if path is not None:
        dp = path_static_distance(points, path, p, max_travel=30.0)
        if dp is not None:
            d = dp
    credit = 0.0
    if nearest is not None:
        for tr in moving_tracks or []:            # is the nearest obstacle a car driving away?
            hx, hy = tr['size'][0] / 2 + 0.5, tr['size'][1] / 2 + 0.5
            if abs(nearest[0] - tr['pos'][0]) < hx and abs(nearest[1] - tr['pos'][1]) < hy and tr['vel'][0] > 0:
                credit = tr['vel'][0] ** 2 / (2 * p.a_lead_max)
                break
    v_static = safe_speed(d + credit, p) if math.isfinite(d) else math.inf
    if cmd_v <= 0.5 and ego_v <= 0.6 and math.isfinite(d) and d > 0.4:
        # MANOEUVRING at walking pace (freeing a wedged corner, parking): stopping takes ~0.2 m,
        # so 0.4 m of free path is enough - the 2 m cruising margin would forbid every manoeuvre
        v_static = max(v_static, min(cmd_v, 0.5))
    ttc = time_to_collision(max(v_plan, 0.5), steer, moving_tracks, p) if moving_tracks else math.inf
    if ttc < p.ttc_stop:
        v_dyn = 0.0
    elif ttc < p.ttc_slow:
        v_dyn = cmd_v * (ttc - p.ttc_stop) / (p.ttc_slow - p.ttc_stop)
    else:
        v_dyn = math.inf
    v = max(0.0, min(cmd_v, v_static, v_dyn))
    reason = 'clear'
    if v < cmd_v - 1e-3:
        reason = 'static_obstacle' if v_static <= v_dyn else 'moving_obstacle'
    return v, {'static_distance': d, 'lead_credit': credit, 'ttc': ttc, 'v_static': v_static,
               'v_dynamic': v_dyn, 'reason': reason}
