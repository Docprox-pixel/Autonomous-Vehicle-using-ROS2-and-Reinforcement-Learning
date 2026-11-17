"""Local planner: choose steering + speed from what the lidars see (pure Python, no ROS).

Every cycle a fan of constant-curvature arcs (hard left .. hard right) is evaluated:

  free      how far the car BODY can travel along the arc before touching an obstacle
            (swept footprint, the same maths the safety shield uses)
  lane      drive in the middle of the RIGHT HALF of the road, measured from the road's two edges
            (kerb/grass/barrier/parked cars from camera + lidar). No painted line is needed; a
            painted centre line, when seen, is an extra hard wall from road_perception.
  smooth    change from the previous steering, and steering magnitude
  goal      progress toward an optional goal point (base_link) - picks left/right at junctions
  clear     closest approach of the body to any obstacle over the first metres (prefer >= 1 m)
  route     with a lane-level route (auto_car_core.lanemap): distance and heading error between
            the arc end and the route path. The route then decides lane and junction paths; the
            lane/goal terms are only used without a route.

The best feasible arc wins. Speed respects the comfort lateral acceleration in curves, the
stopping distance to the arc's first obstacle, and a CORNER PREVIEW: when the road straight
ahead ends within corner_lookahead, the car slows toward v_turn before it starts turning
(a car that enters a corner too fast swings wide into the other lane). Turns are therefore taken because the road
geometry makes a curved arc the longest free one, not because a line or a map says so.
"""
import math
from dataclasses import dataclass

import numpy as np

from auto_car_core.safety import CAR_FRONT, CAR_REAR, HALF_WIDTH, Params, path_poses, safe_speed, static_distance


@dataclass
class PlannerParams:
    n_arcs: int = 31
    max_steer: float = 0.45
    horizon: float = 45.0       # m of arc evaluated (long plans: stable speed, early slowing)
    step: float = 0.5           # m between footprint poses (planner; the shield uses 0.25)
    min_free: float = 6.0       # m: arcs shorter than this are not drivable
    v_max: float = 10.0         # m/s on town roads (36 km/h; road limit 50 km/h)
    v_max_highway: float = 27.8 # m/s on roads the lane map marks as highway (bounded by what is visible)
    a_decel: float = 1.5        # m/s^2 comfortable braking toward an upcoming curve
    far_look: float = 120.0     # m of route corridor checked for obstacles at highway speed
    a_lat: float = 1.2          # m/s^2 comfortable lateral acceleration (dry run: 1.5 left the lane in hairpins)
    v_turn: float = 3.0         # m/s target speed for corners/junction turns
    v_clear_min: float = 1.5    # m/s when the chosen path passes obstacles at the hard margin ...
    clear_full: float = 1.8     # ... rising linearly to unrestricted at this clearance (m)
    corner_lookahead: float = 25.0  # m: straight road shorter than this = corner coming
    clear_target: float = 1.0   # m preferred body-to-obstacle clearance
    clear_min: float = 0.8      # m: passing a REAL obstacle closer than this is penalised (soft; a hard
                                #    tier that also counted the virtual road edge made the car skip turns)
    w_close: float = 1.0
    clear_horizon: float = 10.0 # m of arc over which clearance is checked
    w_clear: float = 0.6
    w_path: float = 1.2          # route following: lateral error at the arc end (capped at 4 m)
    w_path_heading: float = 0.6  # route following: heading error at the arc end
    w_pursuit: float = 1.5       # route following: prefer the arc nearest the pure-pursuit steering
    w_lane_exit: float = 3.0     # per metre the arc end leaves the lane (> 1 m off the lane centre):
                                 # follow a slower car instead of swerving out of a single lane
    pursuit_base: float = 2.5    # m: pure-pursuit look-ahead = base + gain * speed (curve speed);
    pursuit_gain: float = 0.5    #    short in turns: a long look-ahead cuts the inside of the corner
    path_look_min: float = 3.0
    path_look_max: float = 15.0
    path_preview: float = 25.0   # m of route ahead checked for upcoming curves (slow down early)
    keep_right: float = 1.7     # m fallback gap car side -> right edge when the road width is unknown
    lane_fraction: float = 0.25 # car centre at this fraction of the road width from the right edge
    min_road_width: float = 5.0
    max_road_width: float = 30.0
    right_window: float = 8.0   # m ahead in which the right edge is measured
    w_free: float = 1.0
    free_cap: float = 18.0      # m: free distance beyond this (~stopping distance at v_max + margin) earns nothing
    w_straight: float = 0.10
    w_change: float = 0.8
    w_lane: float = 0.15        # per metre of keep-right error (capped)
    w_goal: float = 0.8
    cell: float = 0.25          # m: obstacle points are thinned to one per cell


def thin(points, cell, max_range):
    """One point per (x, y) cell within range: keeps geometry, cuts compute 10x."""
    if points is None or len(points) == 0:
        return np.zeros((0, 3))
    p = points[np.hypot(points[:, 0], points[:, 1]) < max_range]
    if len(p) == 0:
        return p
    # keep the LOWEST point of each cell: with the car's height profile (low hood, taller cabin)
    # the lowest part of a pole/sign is the one the bumper meets first
    p = p[np.argsort(p[:, 2], kind='stable')]
    _, idx = np.unique(np.floor(p[:, :2] / cell).astype(np.int32), axis=0, return_index=True)
    return p[idx]


def side_clearances(points, steer, s, p_safe, max_lateral=25.0):
    """(right gap, left gap) between the car's sides and the nearest edge/obstacle, at travel s."""
    if len(points) == 0:
        return math.inf, math.inf
    ex, ey, eyaw = path_poses(1.0, steer, p_safe, s)          # v = 1 -> t = s
    c, sn = math.cos(eyaw), math.sin(eyaw)
    dx, dy = points[:, 0] - ex, points[:, 1] - ey
    bx, by = c * dx + sn * dy, -sn * dx + c * dy
    band = (bx > CAR_REAR) & (bx < CAR_FRONT + 6.0) & (np.abs(by) < max_lateral)
    right, left = band & (by < -HALF_WIDTH), band & (by > HALF_WIDTH)
    rc = float((-by[right] - HALF_WIDTH).min()) if right.any() else math.inf
    lc = float((by[left] - HALF_WIDTH).min()) if left.any() else math.inf
    return rc, lc


def right_clearance(points, steer, s, p_safe):
    return side_clearances(points, steer, s, p_safe)[0]


def lane_target(rc, lc, width_hint, pp):
    """Desired right gap: middle of the right half of the road, from its measured width.

    Returns (target right gap, measured road width or None)."""
    if math.isfinite(rc) and math.isfinite(lc):
        width = rc + lc + 2 * HALF_WIDTH
        if pp.min_road_width <= width <= pp.max_road_width:
            return max(pp.lane_fraction * width - HALF_WIDTH, 0.5), width
    if width_hint:                                   # edges out of view (junction): remembered width
        return max(pp.lane_fraction * width_hint - HALF_WIDTH, 0.5), None
    return pp.keep_right, None


def min_clearance(points, steer, ps, s_max, step=0.5):
    """Smallest distance between the car body and any obstacle point along the first s_max m."""
    if len(points) == 0:
        return math.inf
    s_arr = np.arange(step, s_max + 1e-9, step)
    poses = np.array([path_poses(1.0, steer, ps, s) for s in s_arr])
    c, sn = np.cos(poses[:, 2])[:, None], np.sin(poses[:, 2])[:, None]
    dx, dy = points[None, :, 0] - poses[:, 0, None], points[None, :, 1] - poses[:, 1, None]
    bx, by = c * dx + sn * dy, -sn * dx + c * dy
    gx = np.maximum(np.maximum(CAR_REAR - bx, bx - CAR_FRONT), 0.0)
    gy = np.maximum(np.abs(by) - HALF_WIDTH, 0.0)
    return float(np.hypot(gx, gy).min())


def path_error(path, ex, ey, eyaw):
    """(lateral distance, heading error) between pose and the nearest point of path (Nx2)."""
    a, b = path[:-1], path[1:]
    ab = b - a
    L2 = np.maximum((ab ** 2).sum(axis=1), 1e-9)
    t = np.clip(((np.array([ex, ey]) - a) * ab).sum(axis=1) / L2, 0.0, 1.0)
    q = a + t[:, None] * ab
    d = np.hypot(q[:, 0] - ex, q[:, 1] - ey)
    i = int(np.argmin(d))
    h = math.atan2(ab[i, 1], ab[i, 0])
    return float(d[i]), abs(math.atan2(math.sin(eyaw - h), math.cos(eyaw - h)))


def path_ahead(path, preview):
    """Part of the path (base_link) from the point nearest the car to `preview` m ahead."""
    d = np.hypot(path[:, 0], path[:, 1])
    i = int(np.argmin(d))
    seg = np.hypot(*np.diff(path[i:], axis=0).T) if len(path) - i > 1 else np.zeros(0)
    j = i + 1 + int(np.searchsorted(np.cumsum(seg), preview))
    return path[i:min(j + 1, len(path))]


def curve_speed_limit(path, a_lat, a_decel, preview=250.0):
    """Fastest speed now that still allows comfortable braking (a_decel) to every curve's speed
    sqrt(a_lat / k) within `preview` m of route: v_now = min_i sqrt(a_lat/k_i + 2 a_decel s_i)."""
    p = path_ahead(path, preview)
    if len(p) < 4:
        return math.inf
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    grid = np.arange(0.0, s[-1], 1.0)
    if len(grid) < 5:
        return math.inf
    pts = np.c_[np.interp(grid, s, p[:, 0]), np.interp(grid, s, p[:, 1])]
    h = np.unwrap(np.arctan2(*np.diff(pts, axis=0).T[::-1]))
    if len(h) < 4:
        return math.inf
    k = np.abs(h[3:] - h[:-3]) / 3.0
    sk = grid[2:2 + len(k)]
    k = np.maximum(k, 1e-6)
    return float(np.min(np.sqrt(a_lat / k + 2 * a_decel * sk)))


def corridor_free(points, path, length, half_width, ps):
    """Distance along the route to the first obstacle point inside the lane corridor (base_link).
    Used at highway speed, beyond the arc planner's horizon."""
    p = path_ahead(path, length)
    if len(p) < 2 or points is None or len(points) == 0:
        return math.inf
    q = points[(points[:, 2] > ps.min_h) & (points[:, 2] < ps.max_h) & (points[:, 0] > CAR_FRONT)]
    if len(q) == 0:
        return math.inf
    a, b = p[:-1], p[1:]
    ab = b - a
    L = np.hypot(*ab.T)
    s0 = np.r_[0.0, np.cumsum(L)][:-1]
    L2 = np.maximum(L ** 2, 1e-9)
    t = np.clip(((q[:, None, :2] - a[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
    proj = a[None] + t[..., None] * ab[None]
    d = np.hypot(*(q[:, None, :2] - proj).transpose(2, 0, 1))
    j = np.argmin(d, axis=1)
    lat = d[np.arange(len(q)), j]
    along = s0[j] + t[np.arange(len(q)), j] * L[j]
    hit = along[lat < half_width]
    return float(hit.min()) if len(hit) else math.inf


def max_curvature(path):
    """Largest curvature (1/m) along a polyline, from turning angle per metre over 3 m chords."""
    if len(path) < 3:
        return 0.0
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    grid = np.arange(0.0, s[-1], 1.0)
    if len(grid) < 4:
        return 0.0
    pts = np.c_[np.interp(grid, s, path[:, 0]), np.interp(grid, s, path[:, 1])]
    h = np.unwrap(np.arctan2(*np.diff(pts, axis=0).T[::-1]))
    if len(h) < 4:
        return 0.0
    dh = np.abs(h[3:] - h[:-3]) / 3.0
    return float(dh.max())


GROUND_Z = -0.32            # base_link z of the road
CAR_HEIGHT = 2.37           # road -> top of the roof lidar (config/sensor_layout.yaml vehicle)
CAR_WIDTH = 2.2             # with mirrors


def passability(points, steer, ps=Params(), look=20.0, step=0.5, n_physical=None):
    """Can the car pass along this arc? Compares what is in its swept corridor with its own size.

    Returns a dict (heights in m above the road, widths in m):
      ceiling      lowest overhead point above the car's path (None = open sky)
      headroom     the car needs: ceiling must be > CAR_HEIGHT + margin
      tallest_low  tallest object in the path that is low enough to drive over (<= drive-over height)
      blocking     the nearest object in the path the car can neither pass over nor under (or None),
                   with its height span and distance
      gap          narrowest free width across the corridor (left + right free space + car width)
      verdict      'pass' | 'too_low' (ceiling) | 'blocked' | 'too_narrow'
    Only REAL (lidar) points are used when n_physical is given."""
    p = points[:n_physical] if n_physical is not None else points
    out = {'ceiling': None, 'headroom': round(ps.max_h - GROUND_Z, 2), 'tallest_low': 0.0, 'blocking': None,
           'gap': None, 'car_width': CAR_WIDTH, 'car_height': CAR_HEIGHT, 'verdict': 'pass'}
    if p is None or len(p) == 0:
        return out
    p = p[np.hypot(p[:, 0], p[:, 1]) < look + CAR_FRONT + 3.0]
    k = math.tan(steer) / ps.wheelbase
    best_gap, lateral = math.inf, False
    for s in np.arange(0.0, look + 1e-9, step):
        x, y, yaw = path_poses(1.0, steer, ps, s)
        c, sn = math.cos(yaw), math.sin(yaw)
        dx, dy = p[:, 0] - x, p[:, 1] - y
        bx, by = c * dx + sn * dy, -sn * dx + c * dy
        ahead = (bx > CAR_FRONT - 0.5) & (bx < CAR_FRONT + step)      # slice just ahead of the bumper
        if not ahead.any():
            continue
        z = p[ahead, 2] - GROUND_Z
        yy = by[ahead]
        inside = np.abs(yy) < HALF_WIDTH + 0.1                         # within the car's width
        over = inside & (z > ps.max_h - GROUND_Z)
        if over.any():
            cz = float(z[over].min())
            out['ceiling'] = cz if out['ceiling'] is None else min(out['ceiling'], cz)
        low = inside & (z <= ps.min_h - GROUND_Z)
        if low.any():
            out['tallest_low'] = max(out['tallest_low'], float(z[low].max()))
        block = inside & (z > ps.min_h - GROUND_Z) & (z <= ps.max_h - GROUND_Z)
        if block.any() and out['blocking'] is None:
            out['blocking'] = {'distance': round(s, 1), 'bottom': round(float(z[block].min()), 2),
                               'top': round(float(z[block].max()), 2)}
        band = (z > ps.min_h - GROUND_Z) & (z <= ps.max_h - GROUND_Z)   # side walls / obstacles
        left = yy[band & (yy > 0)]
        right = yy[band & (yy < 0)]
        if len(left) and len(right):
            g = float(left.min() - right.max())
            if g < best_gap:
                best_gap, lateral = g, left.min() > 0.7 and right.max() < -0.7   # walls at the sides
    if math.isfinite(best_gap):
        out['gap'] = round(best_gap, 2)
    if out['ceiling'] is not None:
        out['ceiling'] = round(out['ceiling'], 2)
    out['tallest_low'] = round(out['tallest_low'], 2)
    if out['blocking'] is not None:
        b = out['blocking']
        out['verdict'] = 'too_low' if b['bottom'] > 1.0 else 'blocked'   # overhang below roof vs object
    if out['gap'] is not None and out['gap'] < CAR_WIDTH + 0.2 and lateral and out['verdict'] != 'too_low':
        out['verdict'] = 'too_narrow'        # walls on both sides closer than the car is wide
    return out


def stanley(path, v, ps, k=1.6, v_soft=1.0, max_steer=0.45):
    """Stanley path tracking (front-axle reference, Thrun et al. 2006): steer = heading error of the
    path at the front axle + atan(k * cross-track error / (v_soft + v)). Follows tight turns without
    cutting corners (pure pursuit / arc-end scoring cut a R = 7 m turn by 2.7 m, over the kerb)."""
    fx = ps.wheelbase                                         # front axle in base_link (rear axle origin)
    d = np.hypot(path[:, 0] - fx, path[:, 1])
    i = int(np.argmin(d))
    j0, j1 = max(i - 1, 0), min(i + 1, len(path) - 1)
    if j1 == j0:
        return None
    t = path[j1] - path[j0]
    t = t / max(np.hypot(*t), 1e-9)
    heading_err = math.atan2(t[1], t[0])                      # car heading is 0 in base_link
    e = float((path[i, 0] - fx) * -t[1] + (path[i, 1]) * t[0])   # + = path left of the front axle
    # project onto the segment for a smooth cross-track error
    a = path[j0]
    ab = path[j1] - a
    tt = np.clip(((np.array([fx, 0.0]) - a) @ ab) / max(ab @ ab, 1e-9), 0, 1)
    q = a + tt * ab
    e = float((q - np.array([fx, 0.0])) @ np.array([-t[1], t[0]]))
    return float(np.clip(heading_err + math.atan2(k * e, v_soft + max(v, 0.0)), -max_steer, max_steer))


EDGE_SHIFT = 0.3    # m: the most a perceived road edge (not an obstacle) may shift the path


def nudge_in_lane(path, points, ps, preview=25.0, max_shift=0.7, prev_shift=None, rate=0.03, road_edges=None):
    """In-lane NUDGE (Apollo / Autoware path nudging): shift the route sideways, at most max_shift
    (the car stays inside its 3.5 m lane: 1.0 m half width + 0.6 < 1.75), away from walls/objects
    that come closer than the shield's corridor (tunnel portal: wall 1.46 m from the lane centre).
    Returns the shifted path (same points, moved along the local left normal)."""
    nudge_in_lane.last = 0.0 if prev_shift is None else float(prev_shift - np.clip(prev_shift, -rate, rate))
    nudge_in_lane.clear = math.inf
    nudge_in_lane.why = ''
    if path is None or len(path) < 3 or points is None or len(points) == 0:
        return path
    pts = points[(points[:, 2] > ps.min_h) & (points[:, 2] < ps.max_h)]
    if len(pts) == 0:
        return path
    A = path_ahead(path, preview)
    if len(A) < 3:
        return path
    t = np.gradient(A, axis=0)
    t /= np.maximum(np.hypot(*t.T), 1e-9)[:, None]
    nl = np.c_[-t[:, 1], t[:, 0]]
    sA = np.r_[0.0, np.cumsum(np.hypot(*np.diff(A, axis=0).T))]
    kappa = np.gradient(np.unwrap(np.arctan2(t[:, 1], t[:, 0])), np.maximum(sA, 1e-6) if len(sA) > 1 else sA)
    kappa = np.convolve(kappa, np.ones(5) / 5.0, mode='same')     # smoothed signed curvature (+ left)
    want = HALF_WIDTH + ps.corridor_margin + 0.35         # keep ~0.75 m from roadside poles/walls
    need_left, need_right = -9.0, -9.0                     # -9: nothing beside the path
    phys_l, phys_r = -9.0, -9.0                            # the same, real obstacles only
    near_need = False                                      # an obstacle needing a shift < 12 m ahead
    # PERCEIVED ROAD EDGES (camera asphalt boundary, lidar kerbs): not only the map route - the body
    # is kept >= 0.3 m inside the road the car actually sees (map lanes can be off the drawn road)
    if road_edges is not None and len(road_edges):
        E = np.asarray(road_edges)[:, :2]
        E = E[np.hypot(E[:, 0], E[:, 1]) < preview + 5.0]
        want_e = HALF_WIDTH + 0.3
        for k in range(0, len(A), 2):
            rel = E - A[k]
            along, lat = rel @ t[k], rel @ nl[k]
            # kerbs / dividers / road edge on the INSIDE of a turn: add the off-tracking
            kk = float(kappa[k])
            # half the off-tracking only: the junction corners now have real kerb returns, and the
            # full allowance against the perceived edge pushed the car 0.5-0.7 m over the centre
            # line in every right turn - into oncoming traffic (drive trace 2026-11-16)
            off_in = 0.5 * min(0.8, ps.wheelbase ** 2 * abs(kk) / 2.0)
            we_r = want_e + (off_in if kk < -1e-3 else 0.0)
            we_l = want_e + (off_in if kk > 1e-3 else 0.0)
            mk = (np.abs(along) < 0.6) & (np.abs(lat) < max(we_r, we_l) + max_shift)
            if not mk.any():
                continue
            r = lat[mk][lat[mk] < 0]
            l_ = lat[mk][lat[mk] > 0]
            # a perceived edge moves the path at most EDGE_SHIFT (the lane map is right to within
            # that now); only real obstacles may use the full max_shift
            if len(r) >= 2:
                need_left = max(need_left, min(EDGE_SHIFT, we_r + float(np.percentile(r, 80))))
            if len(l_) >= 2:
                need_right = max(need_right, min(EDGE_SHIFT, we_l - float(np.percentile(l_, 20))))
    for k in range(0, len(A), 2):
        rel = pts[:, :2] - A[k]
        along, lat = rel @ t[k], rel @ nl[k]
        m = (np.abs(along) < 0.6) & (np.abs(lat) < want + max_shift + 0.5)
        if not m.any():
            continue
        # above the roof only the sensor mount is that wide: high points count as if further out
        from auto_car_core.safety import ROOF_Z, MOUNT_HALF_WIDTH, CABIN_Z, CABIN_HALF_WIDTH
        hi = pts[:, 2] > ROOF_Z
        cab = (pts[:, 2] > CABIN_Z) & ~hi
        lat = np.where(hi, lat + np.sign(lat) * (HALF_WIDTH - MOUNT_HALF_WIDTH),
                       np.where(cab, lat + np.sign(lat) * (HALF_WIDTH - CABIN_HALF_WIDTH), lat))
        r = lat[m][lat[m] < 0]
        l_ = lat[m][lat[m] > 0]
        # OFF-TRACKING in a tight corner: the car's inner side sweeps inside the path by about
        # wheelbase^2 / (2 R) (0.7 m in a 6.6 m hairpin) - keep that much more from posts / kerb
        # corners on the INSIDE of the turn (the front corner met a sign post on the island)
        kk = float(kappa[k])
        off_in = min(0.8, ps.wheelbase ** 2 * abs(kk) / 2.0)
        want_r = want + (off_in if kk < -1e-3 else 0.0)              # right turn: inside = right
        want_l = want + (off_in if kk > 1e-3 else 0.0)               # left turn: inside = left
        if (len(r) and want_r + float(r.max()) > 0.05) or (len(l_) and want_l - float(l_.min()) > 0.05):
            near_need = near_need or sA[k] < 12.0
        if len(r):
            need_left = max(need_left, want_r + float(r.max()))     # wall on the right: move left
            phys_l = max(phys_l, want_r + float(r.max()))
        if len(l_):
            need_right = max(need_right, want_l - float(l_.min()))   # object on the left: move right
            phys_r = max(phys_r, want_l - float(l_.min()))
    # side clearance of the car body along the path (after the shift), for the tight-space speed
    # tight-space speed: REAL obstacles only (walls, poles, cars) - road edges shift the path but
    # must not slow the car (open road looked like a narrow passage: 3 m/s everywhere)
    nudge_in_lane.clear = max(0.0, want - max(phys_l, phys_r) - ps.corridor_margin - 0.35) \
        if (phys_l > -9 or phys_r > -9) else math.inf
    shift = float(np.clip(max(need_left, 0.0) - max(need_right, 0.0), -max_shift, max_shift))
    # what drives the shift (for the drive trace): physical obstacle or perceived road edge, side
    if abs(shift) > 0.02:
        side = 'R' if shift > 0 else 'L'                   # the side the cause is on
        phys = phys_l if shift > 0 else phys_r
        nudge_in_lane.why = f'{"obj" if phys >= abs(shift) - 0.05 else "edge"}{side}'
    if prev_shift is not None:
        # smooth and persistent: the path moves sideways at most `rate` per cycle (0.3 m/s), so a
        # lidar point appearing/disappearing never makes the path (and the steering) jump;
        # twice as fast when the obstacle is close (< 12 m) and the shift must be there in time
        r_ = rate * (2.0 if near_need else 1.0)
        shift = float(prev_shift + np.clip(shift - prev_shift, -r_, r_))
    nudge_in_lane.last = shift
    if abs(shift) < 0.02:
        return path
    t_all = np.gradient(path, axis=0)
    t_all /= np.maximum(np.hypot(*t_all.T), 1e-9)[:, None]
    return path + shift * np.c_[-t_all[:, 1], t_all[:, 0]]


def signed_offset(path, xs, ys):
    """Signed lateral distance (+ left) of points (xs, ys) from the polyline `path` (base_link)."""
    a, ab = path[:-1], np.diff(path, axis=0)
    L2 = np.maximum((ab ** 2).sum(1), 1e-9)
    X, Y = np.asarray(xs, float)[:, None], np.asarray(ys, float)[:, None]
    t = np.clip(((X - a[:, 0]) * ab[:, 0] + (Y - a[:, 1]) * ab[:, 1]) / L2, 0.0, 1.0)
    qx, qy = a[:, 0] + t * ab[:, 0], a[:, 1] + t * ab[:, 1]
    d = np.hypot(X - qx, Y - qy)
    k = np.argmin(d, axis=1)
    r = np.arange(len(k))
    sg = np.sign(ab[k, 0] * (Y[:, 0] - qy[r, k]) - ab[k, 1] * (X[:, 0] - qx[r, k]))
    return sg * d[r, k]


def lane_boundary(path, steer, v, ps, prev_steer=0.0, bound=0.75, horizon=1.5, lag=0.5, max_steer=0.45):
    """HARD LANE BOUNDARY (lane-keeping barrier): predict the car body centre over the next
    `horizon` s with this steering (first-order steering lag from the current steering) and, if it
    would end up more than `bound` from the lane centre (0.75 m: a 2.0 m body in a 3.5 m lane),
    replace the steering by the closest one that keeps it inside. Returns (steer, outside_now)."""
    if path is None or len(path) < 3:
        return steer, False
    P = path_ahead(path, max(v, 2.0) * horizon + 12.0)
    if len(P) < 3:
        return steer, False
    e_now = float(signed_offset(P, [1.45], [0.0])[0])            # body centre now
    dt, n = 0.15, int(horizon / 0.15)
    vv = max(v, 1.5)

    def predict(st):
        x = y = yaw = 0.0
        cur = prev_steer
        xs, ys = [], []
        for _ in range(n):
            cur += (st - cur) * min(1.0, dt / lag)
            x += vv * math.cos(yaw) * dt
            y += vv * math.sin(yaw) * dt
            yaw += vv * math.tan(cur) / ps.wheelbase * dt
            xs.append(x + 1.45 * math.cos(yaw))
            ys.append(y + 1.45 * math.sin(yaw))
        return float(np.max(np.abs(signed_offset(P, xs, ys))))

    limit = max(bound, abs(e_now) + 0.05)                          # outside already: must not get worse
    if predict(steer) <= limit:
        return steer, abs(e_now) > bound
    best, best_err = steer, math.inf
    for st in sorted(np.arange(-max_steer, max_steer + 1e-9, 0.05), key=lambda c: abs(c - steer)):
        err = predict(float(st))
        if err <= limit:
            return float(st), abs(e_now) > bound
        if err < best_err:
            best, best_err = float(st), err
    return best, abs(e_now) > bound


def track_route(path, v, ps, max_steer=0.45):
    """Lane following on the route with a CONNECTOR (as Werling-type planners do): a smooth cubic
    curve from the car's current pose AND heading that blends onto the route 8-20 m ahead (longer
    at speed), followed by pure pursuit. Aiming straight at the route ignored the car's heading:
    with the real steering lag (~1 s measured) the car overshot, swung wider each time and ended
    against the tunnel wall. Tested with 0.3-1.2 s steering lag (test/test_arc_tracking.py)."""
    if path is None or len(path) < 2:
        return None
    Lb = max(5.0, 1.8 * max(v, 0.0) + 4.0)      # (audit of all 122 turns: 8 m blend drifted in tight turns)
    P = path_ahead(path, Lb + 4.0)
    if len(P) < 3:
        return None
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    sb = min(Lb, s[-1] - 0.5)
    if sb < 2.0:
        return None
    tx, ty = np.interp(sb, s, P[:, 0]), np.interp(sb, s, P[:, 1])
    s2 = min(sb + 1.0, s[-1])
    tx2, ty2 = np.interp(s2, s, P[:, 0]), np.interp(s2, s, P[:, 1])
    hT = math.atan2(ty2 - ty, tx2 - tx)
    d = math.hypot(tx, ty)
    c1 = np.array([d / 3.0, 0.0])
    c2 = np.array([tx, ty]) - d / 3.0 * np.array([math.cos(hT), math.sin(hT)])
    u = np.linspace(0.0, 1.0, 60)[:, None]
    B = 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * np.array([tx, ty])
    B = np.vstack([B, P[np.searchsorted(s, sb):]])
    sB = np.r_[0.0, np.cumsum(np.hypot(*np.diff(B, axis=0).T))]
    L = max(2.5, 2.0 + 0.25 * max(v, 0.0))
    ax, ay = np.interp(L, sB, B[:, 0]), np.interp(L, sB, B[:, 1])
    ld2 = ax * ax + ay * ay
    if ld2 < 1.0:
        return None
    st = math.atan(2.0 * ps.wheelbase * ay / ld2)
    # STRAIGHT ROAD, car already in its lane (< 0.2 m, < 2 deg): tight, calm tracking - half gain,
    # so tiny errors give tiny corrections instead of a left-right zig-zag at any speed
    i0 = int(np.argmin(np.hypot(P[:, 0], P[:, 1])))
    lat_err = abs(float(P[i0, 1]))
    j = min(i0 + 3, len(P) - 1)
    head_err = abs(math.atan2(P[j, 1] - P[i0, 1], max(P[j, 0] - P[i0, 0], 1e-3)))
    straight = max_curvature(path_ahead(path, 30.0)) < 0.01
    if straight and lat_err < 0.2 and head_err < math.radians(2.0):
        st *= 0.5
    return float(np.clip(st, -max_steer, max_steer))


def plan(obstacle_points, prev_steer=0.0, goal=None, pp=PlannerParams(), ps=Params(), road_width=None,
         path=None, n_physical=None, highway=False, ego_v=None, track=True, avoid=False, raw_points=None,
         nudge_prev=None, road_edges=None):
    """Return (steer, speed, info). obstacle_points: Nx3 in base_link. goal: (x, y) in base_link.
    path: optional lane-level route (Nx2, base_link) from auto_car_core.lanemap."""
    route = path is not None and len(path) >= 2
    # Height band FIRST, then thin to one point per cell. The other order lost poles: a pole returns
    # points stacked in one (x, y) cell at many heights; if the point kept by thinning was above the
    # roof or near the ground, the band filter then removed it and the pole vanished (the planner
    # steered at a traffic-light pole the shield kept blocking: a deadlock).
    if obstacle_points is not None and len(obstacle_points):
        band_mask = (obstacle_points[:, 2] > ps.min_h) & (obstacle_points[:, 2] < ps.max_h)
        op = obstacle_points[band_mask]
        # physical (lidar) vs virtual (camera road edge, map lane walls, rule walls) points
        is_phys = np.zeros(len(obstacle_points), bool)
        is_phys[:len(obstacle_points) if n_physical is None else n_physical] = True
        op_phys, op_virt = obstacle_points[band_mask & is_phys], obstacle_points[band_mask & ~is_phys]
    else:
        op = op_phys = op_virt = np.zeros((0, 3))
    pts = thin(op, pp.cell, pp.horizon + CAR_FRONT + 6.0)
    pts_phys = thin(op_phys, pp.cell, pp.horizon + CAR_FRONT + 6.0)
    pts_virt = thin(op_virt, pp.cell, pp.horizon + CAR_FRONT + 6.0)

    def free_along(st, max_travel, step, exact=False):
        """Free distance: REAL obstacles with the full contact rule; VIRTUAL walls never mean
        'contact now' (a lane line beside the car must not freeze it - it has to be able to steer
        back into its lane), they only end the arc where the car would cross them."""
        a_ = op_phys if exact else pts_phys
        b_ = op_virt if exact else pts_virt
        return min(static_distance(a_, st, ps, max_travel=max_travel, step=step),
                   static_distance(b_, st, ps, max_travel=max_travel, step=step, contact=False))
    # Clearance uses the UNTHINNED points near the car: a thinned cell keeps one arbitrary point, which
    # overstated clearance by up to ~0.3 m; the planner then cut a junction corner the shield (which
    # sees every point) treated as contact, and the car froze against it.
    near = op[np.hypot(op[:, 0], op[:, 1]) < pp.clear_horizon + CAR_FRONT + 3.0] if len(op) else op
    near = thin(near, 0.05, pp.clear_horizon + CAR_FRONT + 3.0) if len(near) else near
    # n_physical: the first n points of obstacle_points are real (lidar) obstacles, the rest virtual
    # (camera road edge, rule walls). Close passes are penalised against REAL obstacles only: the
    # virtual kerb is close on every arc of a right turn, and counting it made the car skip turns.
    if n_physical is None:
        phys = near
    else:
        po = obstacle_points[:n_physical]
        po = po[(po[:, 2] > ps.min_h) & (po[:, 2] < ps.max_h)] if len(po) else np.zeros((0, 3))
        po = po[np.hypot(po[:, 0], po[:, 1]) < pp.clear_horizon + CAR_FRONT + 3.0] if len(po) else po
        phys = thin(po, 0.05, pp.clear_horizon + CAR_FRONT + 3.0) if len(po) else po
    if route and track:
        # LANE FOLLOWING FIRST: steer along the route with Stanley; keep it if the physical world
        # allows it, otherwise fall through to the arc planner (avoidance).
        v_now = ego_v if ego_v is not None else pp.v_turn
        lane_path = path                                    # the map lane: the hard boundary's reference
        path = nudge_in_lane(path, op_phys, ps, prev_shift=nudge_prev, road_edges=road_edges)
        nudge_now = nudge_in_lane.last
        st_tr = track_route(path, v_now, ps, max_steer=pp.max_steer)
        outside_lane = False
        if st_tr is not None:
            st_tr, outside_lane = lane_boundary(lane_path, st_tr, v_now, ps, prev_steer=prev_steer,
                                                bound=0.75 + abs(nudge_now))
        if st_tr is not None:
            phys_free = min(static_distance(op_phys, st_tr, ps, max_travel=pp.horizon, step=0.25), pp.horizon)
            lane_free = min(corridor_free(op_phys, path, pp.far_look if highway else pp.horizon,
                                          HALF_WIDTH + 0.3, ps), pp.far_look if highway else pp.horizon)
            # ROUTE FIRST: with a route the car steers ONLY along the remembered route; lidar never
            # steers it (no reacting to every lidar ripple). Something in the lane only lowers the
            # speed - down to stopping before it. Avoiding (leaving the route line with the arcs) is
            # allowed only when the caller says the lane has been blocked for a while (avoid=True).
            if not avoid:
                # agree with the safety shield (same exact check on the same steering arc), or the two
                # deadlock: near a wall (tunnel) the shield saw the constant arc of a small correction
                # run into the wall and stopped the car. Take the steering closest to the route's that
                # the shield accepts.
                # EXACTLY the shield's rule (safety.evaluate): same raw lidar points, same
                # static_distance, same safe_speed
                shield_pts = raw_points if raw_points is not None else op_phys
                shield_v = lambda st_: safe_speed(static_distance(shield_pts, st_, ps), ps)
                want = max(1.0, min(ego_v or 0.0, 3.0))
                from auto_car_core.safety import path_static_distance
                d_path = path_static_distance(shield_pts, path, ps, max_travel=30.0)
                if d_path is not None:
                    # the shield checks the same planned path: no steering search needed
                    shield_v = lambda st_, _d=d_path: safe_speed(_d, ps)
                elif shield_v(st_tr) < want:
                    # the search only asks "can it creep at `want`?": look just past that stopping distance
                    reach = want * ps.latency + want * want / (2 * ps.a_brake) + ps.margin + 1.0
                    shield_v = lambda st_, _r=reach: safe_speed(static_distance(shield_pts, st_, ps, max_travel=_r), ps)
                    best_c = (shield_v(st_tr), st_tr)
                    # small corrections only, and never further left when the car is already left of
                    # its route (the search walked the car over the centre line in the tunnel):
                    # rather slow down / stop in the own lane than change lanes
                    i_n = int(np.argmin(np.hypot(path[:, 0], path[:, 1])))
                    left_of_route = path[i_n, 1] < -0.4
                    for dst in (0.05, -0.05, 0.1, -0.1):
                        if dst > 0 and left_of_route:
                            continue
                        cand = float(np.clip(st_tr + dst, -pp.max_steer, pp.max_steer))
                        vc_ = shield_v(cand)
                        if vc_ >= want:
                            best_c = (vc_, cand)
                            break
                        best_c = max(best_c, (vc_, cand))
                    st_tr = best_c[1]
                # look far enough ahead to brake comfortably from the current speed to a sharp
                # corner's speed (25 m was too short above 30 km/h: arrived at a hairpin at 21 km/h)
                v_now_ = max(ego_v or 0.0, 0.0)
                preview = max(pp.path_preview, v_now_ * v_now_ / (2 * pp.a_decel) + 10.0)
                k_ahead = max_curvature(path_ahead(path, preview))
                v_top = pp.v_max_highway if highway else pp.v_max
                # SPEED FROM THE SURROUNDINGS (like a driver), the lowest of:
                #  * curves: v = sqrt(a_lat * radius) at every curve ahead, braking gently (a_decel)
                #    beforehand, so the car is already slow when it reaches a sharp turn
                #  * traffic: 2 s time gap to whatever is ahead in the lane (+ 4 m standstill gap)
                #  * stopping distance (safety) and tight spaces (walls/objects close beside the path)
                v_curve = curve_speed_limit(path, pp.a_lat, pp.a_decel)
                v_follow = max(0.0, (lane_free - 4.0) / 2.0)
                clear = getattr(nudge_in_lane, 'clear', math.inf)
                v_narrow = 3.0 + 6.0 * max(0.0, clear - 0.3) if math.isfinite(clear) else v_top
                # sharpest turn within the preview: be at its speed before reaching it
                v_sharp = max(pp.v_turn, math.sqrt(pp.a_lat / k_ahead)) if k_ahead > 1e-3 else v_top
                parts = {'top': v_top if not outside_lane else 3.0, 'curve': v_curve, 'sharp': v_sharp,
                         'follow': v_follow if lane_free < pp.horizon else v_top, 'narrow': v_narrow,
                         'stop_dist': safe_speed(lane_free, ps)}
                v_tr = min(parts.values())
                parts['shield'] = shield_v(st_tr)
                v_tr = min(v_tr, parts['shield'])                    # never faster than the shield allows
                limit = min(parts, key=parts.get)
                ahead = path_ahead(path, max(lane_free, 10.0))
                return st_tr, max(0.0, v_tr), {
                    'reason': 'ok_lane_follow' if lane_free >= pp.min_free else 'lane_blocked',
                    'arcs': [], 'best': {'steer': st_tr, 'free': lane_free, 'x': ahead[:, 0], 'y': ahead[:, 1]},
                    'free': lane_free, 'road_width': road_width, 'nudge': nudge_now, 'nudge_why': getattr(nudge_in_lane, 'why', ''), 'limit': limit}
            if phys_free >= pp.min_free and lane_free >= pp.min_free:
                k_ahead = max_curvature(path_ahead(path, pp.path_preview))
                v_curve_tr = curve_speed_limit(path, pp.a_lat, pp.a_decel)
                v_top = pp.v_max_highway if highway else pp.v_max
                v_tr = min(v_top, v_curve_tr, max(pp.v_turn, math.sqrt(pp.a_lat / k_ahead)) if k_ahead > 1e-3 else v_top,
                           safe_speed(lane_free, ps))
                ahead = path_ahead(path, max(lane_free, 10.0))     # the path the car will drive (shown in RViz)
                info = {'reason': 'ok_lane_follow', 'arcs': [], 'best': {'steer': st_tr, 'free': lane_free,
                                                                          'x': ahead[:, 0], 'y': ahead[:, 1]},
                        'free': lane_free, 'road_width': road_width}
                return st_tr, max(0.0, v_tr), info
    steers = np.linspace(-pp.max_steer, pp.max_steer, pp.n_arcs)
    # Free distance ALONG THE ROUTE (lane corridor). A constant-curvature arc that steers back to
    # the lane centre would leave the lane on the other side if held for 30 m, so it scored a short
    # 'free' and lost to the arc drifting away from the centre (the car crossed the centre line
    # next to roadside objects). Arcs that follow the route get credit for the route's free length.
    free_route = corridor_free(pts, path, pp.horizon, HALF_WIDTH + 0.2, ps) if route else None
    steer_pp = None
    look_hi = pp.path_look_max
    if route:
        # how far ahead arcs are scored against the route: short in turns (a long look-ahead scores a
        # chord across the curve and the inner wheels cut over the kerb: 2.7 m measured in a R = 7 m
        # turn), long on straight road (stable at speed)
        k15_ = max_curvature(path_ahead(path, 15.0))
        v_here_ = min(pp.v_max, math.sqrt(pp.a_lat / k15_)) if k15_ > 1e-3 else pp.v_max
        look_hi = min(pp.path_look_max, max(5.0, 2.0 + 0.8 * v_here_))
        # Pure pursuit on the lane centre: steer toward the route point `look` metres ahead.
        # (Constant-curvature arcs scored only at their end let the car drift up to ~2 m in a
        # 2.5 m lane; pure pursuit is the standard path tracker and keeps it centred.)
        # look-ahead from the speed the car will have here: curve speed for the sharpest bend within
        # 15 m (a long look-ahead cuts tight junction corners), the speed cap on straight road
        k15 = max_curvature(path_ahead(path, 15.0))
        v_here = min(pp.v_max, math.sqrt(pp.a_lat / k15)) if k15 > 1e-3 else pp.v_max
        look_pp = pp.pursuit_base + pp.pursuit_gain * v_here
        ahead = path_ahead(path, look_pp + 2.0)
        if len(ahead) >= 2:
            s_ = np.r_[0.0, np.cumsum(np.hypot(*np.diff(ahead, axis=0).T))]
            tx, ty = np.interp(look_pp, s_, ahead[:, 0]), np.interp(look_pp, s_, ahead[:, 1])
            ld2 = tx * tx + ty * ty
            if ld2 > 1.0 and tx > 0.0:
                steer_pp = float(np.clip(math.atan(2.0 * ps.wheelbase * ty / ld2), -pp.max_steer, pp.max_steer))
    arcs = []
    for st in steers:
        free = min(free_along(float(st), pp.horizon, pp.step), pp.horizon)
        clr = min_clearance(near, float(st), ps, min(pp.clear_horizon, max(free, pp.step)))
        clr_phys = clr if phys is near else min_clearance(phys, float(st), ps, min(pp.clear_horizon, max(free, pp.step)))
        arc = {'steer': float(st), 'free': free, 'clear': clr,
               # hard margin to PHYSICAL obstacles only: a virtual lane line next to the car is a
               # boundary to steer away from, not something the car can collide with
               'feasible': free >= pp.min_free and clr_phys >= ps.hard_margin}
        rc, lc = side_clearances(pts, float(st), min(pp.right_window, max(free - 1.0, 0.0)), ps)
        target, width = lane_target(rc, lc, road_width, pp)
        arc['right_clear'], arc['left_clear'], arc['road_width'], arc['lane_target'] = rc, lc, width, target
        lane_err = min(abs(rc - target), 6.0) if math.isfinite(rc) else 3.0
        # Free road only matters up to what is needed to stop; beyond that the lane term decides.
        # (Otherwise any arc that would eventually meet the road edge loses to straight ahead, and
        # on a narrow road the car never moves over to its own lane.)
        score = (pp.w_free * min(free, pp.free_cap) / pp.free_cap - pp.w_straight * abs(st) / pp.max_steer
                 - pp.w_change * abs(st - prev_steer) / pp.max_steer
                 - pp.w_clear * max(0.0, pp.clear_target - clr) / pp.clear_target
                 - pp.w_close * max(0.0, pp.clear_min - clr_phys) / pp.clear_min)
        if route:
            look = min(max(free, pp.path_look_min), look_hi)
            ex, ey, eyaw = path_poses(1.0, float(st), ps, look)
            d_path, d_head = path_error(path, ex, ey, eyaw)
            arc['path_error'] = d_path
            if d_path < 0.75 and free >= pp.min_free and free_route is not None and free_route > free:
                credit = min(free_route, pp.horizon)
                score += pp.w_free * (min(credit, pp.free_cap) - min(free, pp.free_cap)) / pp.free_cap
                arc['route_free'] = credit
            score -= pp.w_path * min(d_path, 4.0) / 4.0 + pp.w_path_heading * d_head / math.pi
            if steer_pp is not None:
                score -= pp.w_pursuit * abs(float(st) - steer_pp) / pp.max_steer
            score -= pp.w_lane_exit * max(0.0, d_path - 1.0)
        else:
            score -= pp.w_lane * lane_err
        if goal is not None and not route:
            look = min(free, 12.0)
            ex, ey, _ = path_poses(1.0, float(st), ps, look)
            d0, d1 = math.hypot(*goal), math.hypot(goal[0] - ex, goal[1] - ey)
            arc['progress'] = (d0 - d1) / max(look, 1e-3)              # 1 = straight at the goal
            score += pp.w_goal * arc['progress']
        arc['score'] = score
        arcs.append(arc)

    feasible = [a for a in arcs if a['feasible']]
    if not feasible and n_physical is not None and len(op_virt):
        # Virtual guidance (camera road edge, map lane lines) must never stop the car when the
        # physical world is clear: re-plan on the real obstacles only, still following the route
        # (lane centre). The shield keeps guarding the physical obstacles.
        st2, v2, info2 = plan(obstacle_points[:n_physical], prev_steer, goal, pp, ps, road_width, path, None, highway)
        info2['reason'] = info2['reason'] + ('_virtual_ignored' if info2['reason'] == 'ok' else '')
        return st2, v2, info2
    if not feasible:
        return prev_steer, 0.0, {'reason': 'no_free_arc', 'arcs': arcs, 'best': None}
    # Confirm the choice with EXACTLY the shield's check (all points, its step): the planner's coarser
    # check (thinned cells, 0.5 m poses) can miss a point the shield sees, and then the shield holds
    # the car while the planner keeps asking for the same arc forever (a deadlock seen at night).
    best = None
    for a in sorted(feasible, key=lambda a: -a['score']):
        exact = free_along(a['steer'], pp.min_free + 2.0, 0.25, exact=True)
        if exact >= pp.min_free:
            best = a
            break
        a['feasible'] = False
        a['free'] = min(a['free'], exact)
    if best is None:
        return prev_steer, 0.0, {'reason': 'no_free_arc', 'arcs': arcs, 'best': None}
    k = abs(math.tan(best['steer'])) / ps.wheelbase
    v_curve = math.sqrt(pp.a_lat / k) if k > 1e-4 else math.inf
    straight = arcs[len(arcs) // 2]['free']                  # middle arc = steering 0
    ramp = (straight - 10.0) / max(pp.corner_lookahead - 10.0, 1e-3)
    v_corner = pp.v_turn + (pp.v_max - pp.v_turn) * min(max(ramp, 0.0), 1.0)
    if abs(best['steer']) > 0.08:      # sharper than a normal road curve (R 40 m ~ 0.074 rad): a turn
        v_corner = min(v_corner, pp.v_turn)
    # Cluttered places (poles, cones, barrier ends at junctions): the closer the chosen path passes
    # to anything, the slower the car goes, like a careful driver threading through a gap.
    f = min(max((best['clear'] - ps.hard_margin) / max(pp.clear_full - ps.hard_margin, 1e-3), 0.0), 1.0)
    v_clutter = pp.v_clear_min + (pp.v_max - pp.v_clear_min) * f
    v_route = math.inf
    if route:                                   # slow down BEFORE curves/turns on the route
        k_ahead = max_curvature(path_ahead(path, pp.path_preview))
        v_route = max(pp.v_turn, math.sqrt(pp.a_lat / k_ahead)) if k_ahead > 1e-3 else math.inf
    v_top, free_v = pp.v_max, max(best['free'], best.get('route_free', 0.0))
    if highway and route:
        # Highway: allowed above the town cap as far as the lane corridor ahead is visibly clear
        # (roof lidar 100 m) and the car can still brake comfortably for every curve ahead.
        v_top = pp.v_max_highway
        if best.get('path_error', 9.0) < 0.75 and best['free'] >= pp.min_free:
            # following the lane: what limits speed is the free length of the LANE ahead (up to
            # far_look, roof lidar 100 m), not the 30 m constant-curvature arc on a curving road
            free_v = max(free_v, min(corridor_free(op, path, pp.far_look, HALF_WIDTH + 0.4, ps), pp.far_look))
        v_route = min(v_route if math.isfinite(v_route) else math.inf,
                      curve_speed_limit(path, pp.a_lat, pp.a_decel)) if v_route < math.inf else \
            curve_speed_limit(path, pp.a_lat, pp.a_decel)
        v_corner = max(v_corner, v_route)            # the route curve limit replaces the arc heuristic
        v_clutter = max(v_clutter, min(v_top, v_route))
    v = min(v_top, v_curve, v_corner, v_clutter, v_route, safe_speed(free_v, ps))
    return best['steer'], max(0.0, v), {'reason': 'ok', 'arcs': arcs, 'best': best, 'v_curve': v_curve,
                                        'road_width': best.get('road_width'),
                                        'v_corner': v_corner, 'v_clutter': v_clutter, 'free': best['free']}
