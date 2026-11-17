"""Frenet-frame trajectory planner (Werling, Ziegler, Kammel, Thrun, ICRA 2010).

The lane route (lane centre line from the HD map) is the REFERENCE LINE. Trajectories are planned in
its coordinates: s = distance along the lane, d = lateral offset (+ = left). Candidates:
  lateral      quintic d(t): from the car's current offset/heading to an end offset d_T inside OUR
               lane (never into the oncoming lane), over T = 2..4 s;
  longitudinal quartic s(t): from the current speed/acceleration to a target speed v_T
               (cruise, slower) - or a stop profile when the lane ahead is blocked.
Each candidate is converted to x-y, checked against
  * the car's turning limit (curvature <= tan(max_steer) / wheelbase),
  * the car's BODY footprint against physical obstacles (lidar) with a safety margin, and against
    virtual walls (camera road edge) with the bare body,
and scored: J = lateral jerk + lateral offset + time + speed deviation + longitudinal jerk.
The cheapest safe trajectory is followed by pure pursuit on the trajectory itself.

Why: an arc planner scores constant-curvature arcs against the route; in tight village turns it cut
the corner by 2.7 m (over the kerb), drifted into the oncoming lane and looped around junctions.
In the Frenet frame 'follow the lane' is the coordinate system itself.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from auto_car_core.safety import CAR_FRONT, CAR_REAR, HALF_WIDTH, Params, safe_speed


@dataclass
class FrenetParams:
    lane_width: float = 3.5          # m (village/city lanes); end offsets stay inside this lane
    side_margin: float = 0.15        # m kept between the car body and the lane edge
    obstacle_margin: float = 0.3     # m around the body against physical obstacles
    horizons: tuple = (2.0, 3.0, 4.0)
    n_offsets: int = 7
    dt: float = 0.1
    v_max: float = 10.0              # m/s town
    v_max_highway: float = 27.8
    v_turn: float = 3.0
    a_lat: float = 1.5               # m/s^2 comfortable lateral acceleration (curve speeds)
    a_decel: float = 1.5             # m/s^2 comfortable braking toward curves
    a_max: float = 1.2               # m/s^2 accel limit of candidates (gentle, no sudden speed-up)
    a_min: float = -3.5              # m/s^2 brake limit of candidates
    k_jerk: float = 0.1
    k_time: float = 0.1
    k_offset: float = 1.5            # cost of an end offset from the lane centre ...
    k_offset_left: float = 4.5       # ... 3x more toward the centre line (keep right, never drift left)
    k_speed: float = 1.0
    k_lat: float = 1.0
    k_lon: float = 1.0
    stanley_k: float = 1.6
    max_steer: float = 0.45
    wheelbase: float = 2.995


# ------------------------------------------------------------------ polynomials
def quintic(x0, v0, a0, x1, v1, a1, T):
    A = np.array([[T ** 3, T ** 4, T ** 5], [3 * T ** 2, 4 * T ** 3, 5 * T ** 4], [6 * T, 12 * T ** 2, 20 * T ** 3]])
    b = np.array([x1 - (x0 + v0 * T + a0 * T ** 2 / 2), v1 - (v0 + a0 * T), a1 - a0])
    c3, c4, c5 = np.linalg.solve(A, b)
    return np.array([x0, v0, a0 / 2, c3, c4, c5])


def quartic(x0, v0, a0, v1, a1, T):
    A = np.array([[3 * T ** 2, 4 * T ** 3], [6 * T, 12 * T ** 2]])
    b = np.array([v1 - (v0 + a0 * T), a1 - a0])
    c3, c4 = np.linalg.solve(A, b)
    return np.array([x0, v0, a0 / 2, c3, c4, 0.0])


def poly_eval(c, t):
    p = c[0] + c[1] * t + c[2] * t ** 2 + c[3] * t ** 3 + c[4] * t ** 4 + c[5] * t ** 5
    v = c[1] + 2 * c[2] * t + 3 * c[3] * t ** 2 + 4 * c[4] * t ** 3 + 5 * c[5] * t ** 4
    a = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t ** 2 + 20 * c[5] * t ** 3
    j = 6 * c[3] + 24 * c[4] * t + 60 * c[5] * t ** 2
    return p, v, a, j


# ------------------------------------------------------------------ reference line
class Reference:
    """Route (base_link) resampled every 0.5 m, from 5 m behind the car's projection forward."""

    def __init__(self, path, ahead=60.0):
        d = np.hypot(path[:, 0], path[:, 1])
        i = int(np.argmin(d))
        seg = np.hypot(*np.diff(path, axis=0).T)
        s_all = np.r_[0.0, np.cumsum(seg)]
        s0 = s_all[i]
        g = np.arange(max(0.0, s0 - 5.0), min(s_all[-1], s0 + ahead) + 1e-9, 0.5)
        if len(g) < 4:
            g = np.linspace(s_all[0], s_all[-1], max(4, len(path)))
        self.x = np.interp(g, s_all, path[:, 0])
        self.y = np.interp(g, s_all, path[:, 1])
        self.s = g - s0                                  # s = 0 at the car's projection (approx.)
        dx, dy = np.gradient(self.x), np.gradient(self.y)
        self.h = np.unwrap(np.arctan2(dy, dx))
        self.k = np.gradient(self.h) / np.maximum(np.hypot(dx, dy), 1e-6)
        # exact projection of the car (base_link origin) onto the reference
        P = np.c_[self.x, self.y]
        a, b = P[:-1], P[1:]
        ab = b - a
        t = np.clip((-a * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
        q = a + t[:, None] * ab
        j = int(np.argmin(np.hypot(*q.T)))
        self.s_car = self.s[j] + t[j] * np.hypot(*ab[j])
        hj = self.h[j]
        self.d_car = float(-q[j] @ np.array([-math.sin(hj), math.cos(hj)]))    # + = car left of the line
        self.psi = float(math.atan2(math.sin(-hj), math.cos(-hj)))              # car heading - line heading
        self.length = self.s[-1] - self.s_car

    def to_xy(self, s, d):
        s = np.clip(s + self.s_car, self.s[0], self.s[-1])
        x = np.interp(s, self.s, self.x)
        y = np.interp(s, self.s, self.y)
        h = np.interp(s, self.s, self.h)
        return x - d * np.sin(h), y + d * np.cos(h), h


def curve_speed_profile(ref, fp, start=0.0):
    """Highest speed at the car that still allows braking (a_decel) to every curve's speed ahead."""
    ks = np.abs(ref.k)
    ahead = ref.s - ref.s_car
    m = ahead > start
    if not m.any():
        return math.inf
    v_curve = np.sqrt(fp.a_lat / np.maximum(ks[m], 1e-4))
    return float(np.min(np.sqrt(v_curve ** 2 + 2 * fp.a_decel * ahead[m])))


# ------------------------------------------------------------------ collision check
def footprint_hits(xs, ys, hs, pts, half_w, front, rear, ignore_near=0.0):
    """Index of the first pose whose body rectangle contains an obstacle point (or -1)."""
    if pts is None or len(pts) == 0:
        return -1
    c, s = np.cos(hs)[:, None], np.sin(hs)[:, None]
    dx, dy = pts[None, :, 0] - xs[:, None], pts[None, :, 1] - ys[:, None]
    bx, by = c * dx + s * dy, -s * dx + c * dy
    inside = (bx > rear) & (bx < front) & (np.abs(by) < half_w)
    if ignore_near > 0:
        # virtual walls already at the car now never block (it must be able to move away from them)
        near0 = (pts[:, 0] > CAR_REAR - ignore_near) & (pts[:, 0] < CAR_FRONT + ignore_near) & \
                (np.abs(pts[:, 1]) < HALF_WIDTH + ignore_near)
        inside &= ~near0[None, :]
    hit = inside.any(axis=1)
    return int(np.argmax(hit)) if hit.any() else -1


# ------------------------------------------------------------------ planner
def follow(xs, ys, v, fp):
    """Steer along a planned trajectory (base_link): pure pursuit to the trajectory point
    L = 2.5 + 0.3 v metres ahead of the rear axle. The trajectory is already smooth and within the
    car's turning limit (it was planned that way), so following it is exact; sensor points never
    steer the car directly - they only decide which trajectories are safe."""
    L = 2.5 + 0.3 * max(v, 0.0)
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(np.c_[xs, ys], axis=0).T))]
    d0 = float(np.hypot(xs[0], ys[0]))                       # trajectory starts at the car
    target = min(L + d0 * 0.0, s[-1])
    tx, ty = np.interp(target, s, xs), np.interp(target, s, ys)
    ld2 = max(tx * tx + ty * ty, 1e-3)
    return float(np.clip(math.atan(2.0 * fp.wheelbase * ty / ld2), -fp.max_steer, fp.max_steer))


def plan_frenet(path, physical, virtual=None, ego_v=0.0, ego_a=0.0, highway=False, fp=FrenetParams(),
                ps=Params(), speed_cap=None):
    """Return (steer, speed, info). path: route in base_link (Nx2); physical/virtual: Nx3 base_link."""
    ref = Reference(path, ahead=max(40.0, 6.0 * max(ego_v, fp.v_turn) + 20.0))
    if ref.length < 2.0:
        return 0.0, 0.0, {'reason': 'route_end', 'trajectories': [], 'best': None}
    band = lambda P: P[(P[:, 2] > ps.min_h) & (P[:, 2] < ps.max_h)] if P is not None and len(P) else np.zeros((0, 3))
    phys, virt = band(physical), band(virtual)
    v_top = (fp.v_max_highway if highway else fp.v_max)
    if speed_cap is not None:
        v_top = min(v_top, speed_cap)
    v_des = max(0.0, min(v_top, curve_speed_profile(ref, fp)))
    # current lateral state (offset, rate, accel) and longitudinal state
    v0 = max(ego_v, 0.0)
    d0, dd0 = ref.d_car, v0 * math.sin(ref.psi)
    d_lim = max(0.05, fp.lane_width / 2 - HALF_WIDTH - fp.side_margin)
    offsets = np.unique(np.r_[np.linspace(-d_lim, d_lim, fp.n_offsets), 0.0])
    cands = []
    for T in fp.horizons:
        t = np.arange(0.0, T + 1e-9, fp.dt)
        for dT in offsets:
            cd = quintic(d0, dd0, 0.0, dT, 0.0, 0.0, T)
            d, d_v, d_a, d_j = poly_eval(cd, t)
            v_reach = min(v_des, v0 + 0.6 * fp.a_max * T)        # reachable (quartic peak ~1.5x mean accel)
            for vT in sorted({v_reach, 0.7 * v_reach, 0.4 * v_reach, 0.0}):
                cs = quartic(0.0, v0, ego_a, vT, 0.0, T)
                s, s_v, s_a, s_j = poly_eval(cs, t)
                # hold vT after T (extend the trajectory so its length covers stopping/following)
                if s_v.min() < -0.05 or s_a.max() > fp.a_max + 1e-6 or s_a.min() < fp.a_min - 1e-6:
                    continue
                x, y, h = ref.to_xy(s, d)
                # geometric heading/curvature of the x-y path
                gx, gy = np.gradient(x), np.gradient(y)
                step = np.hypot(gx, gy)
                hh = np.unwrap(np.arctan2(gy, gx)) if step.max() > 0.05 else h
                hh = np.where(step > 0.02, hh, h)
                kap = np.abs(np.gradient(hh)) / np.maximum(step, 0.02)
                moving = s_v > 0.3
                if moving.any() and kap[moving].max() > math.tan(fp.max_steer) / fp.wheelbase * 1.05:
                    continue                                   # tighter than the car can turn
                k_off = fp.k_offset_left if dT > 0 else fp.k_offset
                j = (fp.k_jerk * np.sum(d_j ** 2) * fp.dt + fp.k_time * T + k_off * dT ** 2)
                jl = (fp.k_jerk * np.sum(s_j ** 2) * fp.dt + fp.k_time * T + fp.k_speed * (v_des - vT) ** 2)
                cands.append({'T': T, 'dT': float(dT), 'vT': float(vT), 'x': x, 'y': y, 'h': hh, 's': s, 'v': s_v,
                              'cost': fp.k_lat * j + fp.k_lon * jl})
    def add_rejoin():
        # LOW-SPEED MODE (Werling et al. 2010, sec. III-D): at low speed the time-based lateral
        # polynomial squeezes the lateral move into almost no distance (impossible curvature: car
        # stands still with "no_safe_trajectory"). Plan lateral motion over ARC LENGTH instead:
        # d(s) quintic over S metres, s(t) from a gentle speed profile.
        # Rejoin paths are built in x-y (cubic Bezier from the car's pose to a point of the lane ahead
        # with the lane's heading): far from the reference on a curve the Frenet frame is distorted
        # (a path at d=-5 m inside a 10 m bend looks twice as tight as it is).
        for S in (6.0, 9.0, 13.0, 18.0, 24.0):
            if S + 6.0 > ref.length:
                continue
            for dT in offsets:
                tx, ty, th = ref.to_xy(np.array([S, S + 6.0]), np.array([dT, dT]))
                tgt = np.array([tx[0], ty[0]]); hT = float(th[0])
                dist = float(np.hypot(*tgt))
                if dist < 2.0 or tgt[0] < 0.5:
                    continue
                u = np.linspace(0.0, 1.0, 80)[:, None]
                B, kb = None, math.inf
                for f0, f1 in ((0.33, 0.33), (0.5, 0.5), (0.25, 0.5), (0.5, 0.25), (0.6, 0.4), (0.4, 0.6), (0.7, 0.7)):
                    Pc = np.array([[0.0, 0.0], [f0 * dist, 0.0],
                                   tgt - f1 * dist * np.array([math.cos(hT), math.sin(hT)]), tgt])
                    Bc = ((1 - u) ** 3) * Pc[0] + 3 * ((1 - u) ** 2) * u * Pc[1] + 3 * (1 - u) * u ** 2 * Pc[2] + u ** 3 * Pc[3]
                    bx, by = np.gradient(Bc[:, 0]), np.gradient(Bc[:, 1])
                    kc = float((np.abs(np.gradient(np.unwrap(np.arctan2(by, bx)))) / np.maximum(np.hypot(bx, by), 1e-3)).max())
                    if kc < kb:
                        B, kb = Bc, kc
                tail_s = np.arange(S + 0.5, min(ref.length, S + 12.0), 0.5)
                txs, tys, _ = ref.to_xy(tail_s, np.full(len(tail_s), dT))
                G = np.vstack([B, np.c_[txs, tys]])
                sg = np.r_[0.0, np.cumsum(np.hypot(*np.diff(G, axis=0).T))]
                gx, gy = np.gradient(G[:, 0]), np.gradient(G[:, 1])
                hh = np.unwrap(np.arctan2(gy, gx))
                kap = np.abs(np.gradient(hh)) / np.maximum(np.hypot(gx, gy), 0.02)
                if kap[sg < S + 2.0].max() > math.tan(fp.max_steer) / fp.wheelbase * 1.0:
                    continue
                for vT in sorted({min(v_des, fp.v_turn), min(v_des, 1.5)}):
                    if vT <= 0.05:
                        continue
                    T = min(2.0 * sg[-1] / max(v0 + vT, 0.3), 30.0)    # time to drive the whole rejoin path
                    cs = quartic(0.0, v0, ego_a, vT, 0.0, min(T, 6.0))
                    t = np.arange(0.0, T + 1e-9, fp.dt)
                    s_, s_v, s_a, s_j = poly_eval(cs, np.minimum(t, min(T, 6.0)))
                    s_ = np.where(t > min(T, 6.0), s_ + (t - min(T, 6.0)) * vT, s_)
                    s_v = np.where(t > min(T, 6.0), vT, s_v)
                    if s_v.min() < -0.05 or s_a.max() > fp.a_max + 1e-6 or s_a.min() < fp.a_min - 1e-6:
                        continue
                    s_ = np.minimum(s_, sg[-1])
                    x = np.interp(s_, sg, G[:, 0]); y = np.interp(s_, sg, G[:, 1]); h = np.interp(s_, sg, hh)
                    k_off = fp.k_offset_left if dT > 0 else fp.k_offset
                    cost = (fp.k_lat * (float(np.sum(kap ** 2)) * 0.5 + k_off * dT ** 2 + 0.3 * S)
                            + fp.k_lon * fp.k_speed * (v_des - vT) ** 2 + 5.0)
                    cands.append({'T': T, 'dT': float(dT), 'vT': float(vT), 'x': x, 'y': y, 'h': h, 's': s_,
                                  'v': s_v, 'cost': cost, 'mode': 'rejoin'})
    if abs(d0) > d_lim + 0.3:
        add_rejoin()                                       # off the lane: rejoin paths too
    cands.sort(key=lambda c: c['cost'])

    def virt_hit(c):
        # virtual boundaries (camera road edge, kerbs, lane lines) are noisy far away and the plan is
        # redone every 0.1 s: check them only over the next ~1.5 s / 6 m, so a far false edge never
        # makes the planner prefer a trajectory that slows down ("stopping to think")
        d = np.r_[0.0, np.cumsum(np.hypot(np.diff(c['x']), np.diff(c['y'])))]
        m = d < max(6.0, 1.5 * v0 + 4.0)
        return footprint_hits(c['x'][m], c['y'][m], c['h'][m], virt, HALF_WIDTH + 0.02, CAR_FRONT, CAR_REAR,
                              ignore_near=0.5) >= 0
    best = None
    hw_phys = HALF_WIDTH + fp.obstacle_margin
    for c in cands:
        if footprint_hits(c['x'], c['y'], c['h'], phys, hw_phys, CAR_FRONT + fp.obstacle_margin,
                          CAR_REAR - fp.obstacle_margin) >= 0:
            continue
        if virt_hit(c):
            continue
        best = c
        break
    if best is None and v0 < 4.0 and not any(c.get('mode') == 'rejoin' for c in cands):
        # nothing safe at low speed: try the arc-length rejoin paths before giving up
        n0 = len(cands)
        add_rejoin()
        for c in sorted(cands[n0:], key=lambda c: c['cost']):
            if footprint_hits(c['x'], c['y'], c['h'], phys, hw_phys, CAR_FRONT + fp.obstacle_margin,
                              CAR_REAR - fp.obstacle_margin) >= 0:
                continue
            if virt_hit(c):
                continue
            best = c
            break
    if best is None:
        # nothing safe: brake, keep steering along the lane centre
        x, y, h = ref.to_xy(np.arange(0.0, 15.0, 0.5), np.zeros(30))
        return follow(x, y, v0, fp), 0.0, {'reason': 'no_safe_trajectory', 'trajectories': cands[:12],
                                                   'best': None, 'free': 0.0}
    # follow the chosen trajectory: steering by Stanley on its path, speed from its profile shortly ahead
    steer = follow(best['x'], best['y'], v0, fp)
    k = min(int(0.4 / fp.dt), len(best['v']) - 1)
    v_cmd = float(best['v'][k])
    # free distance along the lane (physical obstacles in the lane corridor) caps the speed
    xs, ys, hs = ref.to_xy(np.arange(0.0, min(ref.length, 120.0), 0.5), np.full(int(min(ref.length, 120.0) / 0.5) + 1, best['dT'])[:len(np.arange(0.0, min(ref.length, 120.0), 0.5))])
    hit = footprint_hits(xs, ys, hs, phys, HALF_WIDTH + 0.2, CAR_FRONT, CAR_REAR)
    free = math.inf if hit < 0 else max(0.0, hit * 0.5 - 0.0)
    v_cmd = min(v_cmd if v_cmd > 0.05 else best['vT'], safe_speed(free, ps), v_des) if free < math.inf else min(max(v_cmd, 0.0), v_des)
    return steer, max(0.0, v_cmd), {'reason': 'ok_frenet', 'trajectories': cands[:12], 'best': {
        'steer': steer, 'free': free if free < math.inf else 99.0, 'dT': best['dT'], 'vT': best['vT'], 'T': best['T'],
        'x': best['x'], 'y': best['y']}, 'free': free if free < math.inf else 99.0, 'v_des': v_des}
