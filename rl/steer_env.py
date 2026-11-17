"""RL STEERING (residual) environment on the REAL routes of the lane map.

Every episode drives one route of worlds/village_net (the planner's path through a junction turn:
last 30 m of the arriving lane + junction link + first 30 m of the leaving lane, through
LaneMap.smooth_kinks - exactly what tools/audit_turns.py checks - or a long plain lane), closed
loop with the REAL planner (auto_car_core.planner.plan) and the car model of the audit: steering
rate 0.08 rad / 0.1 s, steering lag 0.15-0.30 s (measured 0.20), yaw gain 0.95, speed from the
planner. The policy adds a residual in [-MAX_RES, MAX_RES] rad to the planner's steering.
Start: random offset +-0.4 m, heading +-4 deg (as after a nudge or a lane change).

Observation: auto_car_core.steer_rl.obs_from_path (same on the car).
Reward per 0.1 s: -lat^2 - 0.3 heading^2 - 40 (d command)^2 (zig-zag) - 0.2 (res/MAX_RES)^2
(no needless corrections); -20 and end if the car is > 1.5 m off the route.
"""
import math
import os
import sys
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'tools'))
import auto_car_core.planner as PL  # noqa: E402
from auto_car_core.lanemap import LaneMap  # noqa: E402
from auto_car_core.steer_rl import MAX_RES, obs_from_path  # noqa: E402

NET = os.path.join(HERE, '..', 'worlds', 'village_net', 'sumo.net.xml')
_PATHS = None


def clip_len(P, length, from_end):
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
    return P[s >= s[-1] - length] if from_end else P[s <= length]


def route_pool(net=NET):
    """The planner's paths for every junction movement + the longer plain lanes."""
    global _PATHS
    if _PATHS is not None:
        return _PATHS
    m = LaneMap(net)
    root = ET.parse(net).getroot()
    off = np.array([float(v) for v in root.find('location').get('netOffset').split(',')])
    lane = {ln.get('id'): np.array([[float(v) for v in q.split(',')[:2]] for q in ln.get('shape').split()]) - off
            for e in root.findall('edge') for ln in e.findall('lane')}
    rev = lambda e: e[1:] if e.startswith('-') else '-' + e
    out = []

    def resample(raw):
        raw = raw[np.r_[True, np.hypot(*np.diff(raw, axis=0).T) > 1e-6]]
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(raw, axis=0).T))]
        ss = np.arange(0.0, s[-1], 1.0)
        return np.c_[np.interp(ss, s, raw[:, 0]), np.interp(ss, s, raw[:, 1])]

    for c in root.findall('connection'):
        f, t, via = c.get('from'), c.get('to'), c.get('via')
        if f.startswith(':') or t == rev(f) or not via or f.startswith('gneE') or t.startswith('gneE'):
            continue
        A, B, V = lane.get(f'{f}_{c.get("fromLane")}'), lane.get(f'{t}_{c.get("toLane")}'), lane.get(via)
        if A is None or B is None or V is None:
            continue
        raw = resample(np.vstack([clip_len(A, 30.0, True), V, clip_len(B, 30.0, False)]))
        if len(raw) >= 20:
            out.append(m.smooth_kinks(raw))
    for lid, L in lane.items():
        if not lid.startswith(':') and not lid.startswith('gneE'):
            R = resample(L)
            if len(R) >= 60:
                out.append(R[:160])
    _PATHS = out
    return out


class SteerEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.paths = route_pool()
        self.pp = PL.PlannerParams()
        self.observation_space = gym.spaces.Box(-3, 3, (9,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.P = self.paths[int(r.integers(len(self.paths)))]
        self.sP = np.r_[0.0, np.cumsum(np.hypot(*np.diff(self.P, axis=0).T))]
        t = self.P[3] - self.P[0]
        h = math.atan2(t[1], t[0])
        e = r.uniform(-0.4, 0.4)
        self.x, self.y = self.P[0] + e * np.array([-math.sin(h), math.cos(h)])
        self.yaw = h + math.radians(r.uniform(-4, 4))
        self.v = r.uniform(2.0, 7.0)
        self.steer = self.cmd = 0.0
        self.lag = r.uniform(0.15, 0.30)
        self.res = 0.0
        self.t = 0
        self._plan()
        return self._obs(), {}

    def _base_path(self):
        c, s_ = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = self.P[:, 0] - self.x, self.P[:, 1] - self.y
        return np.c_[c * dx + s_ * dy, -s_ * dx + c * dy]

    def _plan(self):
        self.rb = self._base_path()
        st, vc, _ = PL.plan(np.zeros((0, 3)), self.cmd, None, pp=self.pp, path=self.rb, n_physical=0, ego_v=self.v)
        self.st_plan, self.vc = float(st), float(vc)

    def _obs(self):
        return obs_from_path(PL.path_ahead(self.rb, 25.0) if len(self.rb) else self.rb, self.v, self.cmd,
                             self.st_plan, self.res)

    def step(self, action):
        dt = 0.1
        self.res = float(np.clip(action[0], -1, 1)) * MAX_RES
        cmd_prev = self.cmd
        self.cmd = float(np.clip(self.st_plan + self.res, self.cmd - 0.08, self.cmd + 0.08))
        self.cmd = float(np.clip(self.cmd, -0.45, 0.45))
        self.steer += (self.cmd - self.steer) * min(1.0, dt / self.lag)
        self.v += float(np.clip(self.vc - self.v, -1.5 * dt, 1.0 * dt))
        c, s_ = math.cos(self.yaw), math.sin(self.yaw)
        self.x += self.v * c * dt
        self.y += self.v * s_ * dt
        self.yaw += 0.95 * self.v * math.tan(self.steer) / 2.995 * dt
        self.t += 1
        i = int(np.argmin(np.hypot(self.P[:, 0] - self.x, self.P[:, 1] - self.y)))
        lo, hi = max(i - 3, 0), min(i + 4, len(self.P))
        lat = abs(float(PL.signed_offset(self.P[lo:hi], [self.x], [self.y])[0])) if hi - lo >= 2 else 0.0
        t = self.P[min(i + 1, len(self.P) - 1)] - self.P[max(i - 1, 0)]
        he = math.atan2(math.sin(self.yaw - math.atan2(t[1], t[0])), math.cos(self.yaw - math.atan2(t[1], t[0])))
        reward = -lat * lat - 0.3 * he * he - 40.0 * (self.cmd - cmd_prev) ** 2 - 0.2 * (self.res / MAX_RES) ** 2
        info = {'lat': lat}
        if lat > 1.5:
            return self._obs(), reward - 20.0, True, False, {**info, 'result': 'off'}
        if i >= len(self.P) - 4 or self.t >= 600:
            return self._obs(), reward, True, False, {**info, 'result': 'end'}
        self._plan()
        return self._obs(), reward, False, False, info
