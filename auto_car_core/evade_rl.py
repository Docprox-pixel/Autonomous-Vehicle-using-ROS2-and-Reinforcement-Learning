"""RL SUDDEN-OBSTACLE handling: brake, swerve within the free road, or both.

Active only while something is IN our lane ahead (lidar points inside the lane corridor on the
route, <= 45 m). The policy (SAC, rl/evade_env.py + rl/train_evade.py) returns an acceleration and a
lateral target for the car centre relative to the lane centre (+ = left, towards the oncoming
lane). The local planner applies the target as a sideways shift of the route it follows and uses
the acceleration only if it brakes HARDER than the speed controller (never to speed up).
Hard envelope (identical to training): target within the road (-0.6 .. +3.5 m); never over the
centre line (> 0.75 m) while an oncoming vehicle would meet us within 4 s; the shield still brakes.
Without the model it returns nothing (today's behaviour: stop before it / in-lane nudge).
"""
import math
import os

import numpy as np

Y_MIN, Y_MAX, CENTRE = -0.6, 3.5, 0.75
A_MIN, A_MAX = -6.0, 1.0
LANE_HALF = 1.75
CORRIDOR = 1.3                      # m from the lane centre: the car's own swept corridor
CAR_FRONT = 4.05
DEFAULT_MODEL = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'rl', 'models',
                                              'evade_sac.zip'))


def to_route_frame(route, xy):
    """(s along the route, l lateral: + left of the route) of points xy (base_link)."""
    a, b = route[:-1], route[1:]
    ab = b - a
    L = np.hypot(*ab.T)
    s0 = np.r_[0.0, np.cumsum(L)][:-1]
    t = np.clip(((xy[:, None, :] - a[None]) * ab[None]).sum(-1) / np.maximum(L ** 2, 1e-9)[None], 0, 1)
    f = a[None] + t[..., None] * ab[None]
    d = np.hypot(*(xy[:, None, :] - f).transpose(2, 0, 1))
    k = np.argmin(d, axis=1)
    i = np.arange(len(xy))
    cross = ab[k, 0] * (xy[:, 1] - a[k, 1]) - ab[k, 1] * (xy[:, 0] - a[k, 0])
    return s0[k] + t[i, k] * L[k], np.sign(cross) * d[i, k]


class Evader:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = None
        self.source = 'off'
        self.y_prev = None
        self.target = 0.0
        self.active = False
        self.dx = math.inf
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import SAC
                self.model = SAC.load(model_path, device='cpu')
                self.source = 'rl'
            except Exception as e:                       # pragma: no cover
                self.error = str(e)

    def observe(self, route, points, tracks, v, a, dt=0.1):
        """route: lane route ahead (base_link, Nx2, from the car); points: obstacle points (Nx3);
        tracks: [(x, y, vx, vy, label)]. Returns (obs, oncoming_meet) or (None, inf) if nothing
        is in our lane ahead."""
        if route is None or len(route) < 3:
            return None, math.inf
        s_car, l_car = to_route_frame(route, np.zeros((1, 2)))
        y = -float(l_car[0])                               # car offset from the lane centre (+ left)
        vy = 0.0 if self.y_prev is None else (y - self.y_prev) / dt
        self.y_prev = y
        R = route[:min(len(route), 60)]
        obs_pts = points[(points[:, 2] > -0.05) & (points[:, 2] < 2.2) & (points[:, 0] > CAR_FRONT - 0.5)] \
            if points is not None and len(points) else np.zeros((0, 3))
        obs_pts = obs_pts[np.hypot(obs_pts[:, 0], obs_pts[:, 1]) < 50.0] if len(obs_pts) else obs_pts
        dx = math.inf
        if len(obs_pts):
            s, l = to_route_frame(R, obs_pts[:, :2])
            # only what BLOCKS the car's own corridor (half width 1.0 + 0.3 m): things at the lane edge
            # (kerb posts, parapets, a parked scooter) are passed with the planner's 0.7 m nudge - the
            # first version reacted to them and swung 3 m into the other lane
            inlane = (np.abs(l) < CORRIDOR) & (s > s_car[0] + CAR_FRONT - 0.5) & (s < s_car[0] + 45.0)
            if inlane.sum() >= 3:
                s_in, l_in = s[inlane], l[inlane]
                s_min = float(np.min(s_in))
                near = s_in < s_min + 3.0
                lo, hi = float(np.min(l_in[near])), float(np.max(l_in[near]))
                dx = s_min - (s_car[0] + CAR_FRONT)
                p_xy = obs_pts[inlane][near][:, :2].mean(0)
        if not math.isfinite(dx):
            return None, math.inf
        # moving objects: the obstacle's own lateral speed / label, and oncoming traffic
        o_vy, ped, meet, nearest = 0.0, 0.0, math.inf, 200.0
        if tracks:
            T = np.array([[t[0], t[1], t[2], t[3]] for t in tracks])
            j = int(np.argmin(np.hypot(T[:, 0] - p_xy[0], T[:, 1] - p_xy[1])))
            if np.hypot(T[j, 0] - p_xy[0], T[j, 1] - p_xy[1]) < 2.5:
                # lateral velocity in the route frame (route roughly along +x near the car)
                o_vy = float(T[j, 3])
                ped = 1.0 if tracks[j][4] == 'person' else 0.0
            ts, tl = to_route_frame(R, T[:, :2])
            for k in range(len(T)):
                if LANE_HALF < tl[k] < 3 * LANE_HALF + 0.5 and T[k, 2] < -1.0:   # other lane, towards us
                    gap = ts[k] - (s_car[0] + CAR_FRONT)
                    if gap > -4.5:
                        meet = min(meet, max(gap, 0.0) / max(v - T[k, 2], 0.5))
                        nearest = min(nearest, gap)
        self.dx = dx
        o = np.array([v / 10, a / 6, y / 2, vy / 2, np.clip(dx, -10, 60) / 30, lo / 2, hi / 2, o_vy / 2,
                      min(meet, 10) / 10, min(nearest, 200) / 100, ped, (y - CENTRE) / 2], np.float32)
        return o, meet

    def step(self, route, points, tracks, v, a, dt=0.1):
        """(active, a_des or None, lateral target, reason)."""
        if self.model is None:
            return False, None, 0.0, 'off'
        o, meet = self.observe(route, points, tracks, v, a, dt)
        if o is None:
            # nothing in the lane: ease back to the lane centre (0.5 m/s)
            self.target = float(np.sign(self.target) * max(abs(self.target) - 0.5 * dt, 0.0))
            self.active = abs(self.target) > 0.05
            return self.active, None, self.target, 'returning' if self.active else 'clear'
        act = self.model.predict(o, deterministic=True)[0]
        a_des = A_MIN + (float(np.clip(act[0], -1, 1)) + 1) * 0.5 * (A_MAX - A_MIN)
        y_t = Y_MIN + (float(np.clip(act[1], -1, 1)) + 1) * 0.5 * (Y_MAX - Y_MIN)
        why = 'rl'
        y_t = float(np.clip(y_t, Y_MIN, Y_MAX))
        if y_t > CENTRE and meet < 4.0 and float(o[2]) * 2 <= CENTRE + 0.05:
            y_t, why = CENTRE, 'envelope: oncoming'
        self.target = y_t
        self.active = True
        return True, a_des, y_t, why
