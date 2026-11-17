"""RL STEERING (residual): small corrections on top of the planner's steering.

The planner (auto_car_core/planner.py track_route) keeps doing the steering; the policy adds at
most +-MAX_RES rad. It learns what the fixed controller cannot: anticipating the 0.2 s steering
lag in turns, settling without the last zig-zag, holding the lane centre on long curves.
Policy: SAC trained in rl/steer_env.py on the real routes of the lane map (rl/train_steer.py).
The same observation builder is used in training and on the car (obs_from_path).
Without the model the residual is 0 (pure planner).
"""
import math
import os

import numpy as np

MAX_RES = 0.05                      # rad: the most the policy may change the planner's steering
DEFAULT_MODEL = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'rl', 'models',
                                              'steer_sac.zip'))


def obs_from_path(path_base, v, steer_now, steer_plan, prev_res):
    """path_base: route ahead in base_link (Nx2, car at origin heading +x)."""
    P = np.asarray(path_base, float)
    if len(P) < 3:
        return np.zeros(9, np.float32)
    d = np.hypot(P[:, 0], P[:, 1])
    i = int(np.argmin(d))
    j = min(i + 1, len(P) - 1)
    k = max(j - 1, 0)
    t = P[j] - P[k]
    head = math.atan2(t[1], t[0]) if np.hypot(*t) > 1e-6 else 0.0
    # signed lateral error: + = route is to the LEFT of the car
    lat = float(P[i, 1]) if abs(P[i, 0]) < 2.0 else float(np.sign(P[i, 1]) * d[i])
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(P[i:], axis=0).T))]
    ahead = []
    for look in (3.0, 8.0, 15.0):
        q = int(min(np.searchsorted(s, look), len(s) - 1)) + i
        q2 = min(q + 1, len(P) - 1)
        tt = P[q2] - P[max(q2 - 1, 0)]
        ahead.append(math.atan2(tt[1], tt[0]) if np.hypot(*tt) > 1e-6 else head)
    return np.array([np.clip(lat, -2, 2), np.clip(head, -1, 1), *np.clip(ahead, -1.6, 1.6), v / 10.0,
                     steer_now / 0.45, steer_plan / 0.45, prev_res / MAX_RES], np.float32)


class SteerResidual:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = None
        self.source = 'planner only'
        self.prev = 0.0
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import SAC
                self.model = SAC.load(model_path, device='cpu')
                self.source = 'rl residual'
            except Exception as e:                       # pragma: no cover
                self.error = str(e)

    def residual(self, path_base, v, steer_now, steer_plan):
        if self.model is None or path_base is None or len(path_base) < 3:
            self.prev = 0.0
            return 0.0
        o = obs_from_path(path_base, v, steer_now, steer_plan, self.prev)
        r = float(np.clip(self.model.predict(o, deterministic=True)[0][0], -1, 1)) * MAX_RES
        self.prev = r
        return r
