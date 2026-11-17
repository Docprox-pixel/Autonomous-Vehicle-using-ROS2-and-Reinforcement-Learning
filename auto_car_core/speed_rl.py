"""RL SPEED CONTROL: how to accelerate / brake towards the planner's target speed.

Policy: SAC trained in rl/speed_env.py (rl/train_speed.py), compared there with the old fixed rule.
The planner still decides the target speed (curves, limits, headway, stop lines); this only
chooses the acceleration. Hard limits stay outside the policy: the speed never exceeds the
target (so a stop before a line or an obstacle is always possible at 1.5 m/s^2), the jerk limiter
of the local planner stays, and the safety shield brakes in emergencies.
Without the model (or stable-baselines3) the old rule is used.
"""
import math
import os

import numpy as np

A_MIN, A_MAX = -3.0, 1.0
DEFAULT_MODEL = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'rl', 'models',
                                              'speed_sac.zip'))


class SpeedController:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = None
        self.source = 'rule'
        self.prev_free = None
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import SAC
                self.model = SAC.load(model_path, device='cpu')
                self.source = 'rl'
            except Exception as e:                       # pragma: no cover
                self.error = str(e)

    @staticmethod
    def observation(v, a, vt, free, rate, room, limit, is_line):
        ttc = free / rate if rate > 0.1 else 99.0
        return np.array([v / 10, a / 3, vt / 10, (vt - v) / 10, min(free, 120) / 60, np.clip(rate / 10, -2, 2),
                         min(ttc, 20) / 10, min(room, 120) / 60, limit / 10, 1.0 if is_line else 0.0], np.float32)

    def accel(self, v, a, vt, free_obj, stop_dist, limit, dt=0.1, firm=False):
        """Desired acceleration (m/s^2). free_obj: free distance to the nearest object in the path
        (m, large if none); stop_dist: distance to a rule stop line (m, large if none)."""
        is_line = stop_dist < free_obj and stop_dist < 1e8
        free = min(free_obj, stop_dist, 200.0)
        # closing speed of what is ahead: a jump of > 2 m between scans is a DIFFERENT object (cross
        # traffic in a junction), not something approaching - the raw difference made the policy
        # brake at full force while the car stood on a green light
        if self.prev_free is None or abs(self.prev_free - free) > 2.0:
            raw = 0.0
        else:
            raw = (self.prev_free - free) / dt
        self.rate = 0.7 * getattr(self, 'rate', 0.0) + 0.3 * raw
        rate = self.rate
        self.prev_free = free
        room = stop_dist if stop_dist < 1e8 else 200.0
        if self.model is not None:
            act = float(self.model.predict(self.observation(v, a, vt, free, rate, room, limit, is_line),
                                           deterministic=True)[0][0])
            a_des = A_MIN + (np.clip(act, -1, 1) + 1) * 0.5 * (A_MAX - A_MIN)
        else:
            a_des = float(np.clip((vt - v) / dt, -3.0 if firm else -1.5, 1.0))
        # hard limit: never above the planner's target speed
        if v + a_des * dt > vt:
            a_des = min(a_des, (vt - v) / dt)
        return float(np.clip(a_des, A_MIN, A_MAX))
