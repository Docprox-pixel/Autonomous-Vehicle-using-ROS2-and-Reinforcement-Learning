"""RL PARKING (goal-conditioned SAC + HER, rl/park_env.py / rl/train_park.py).

Bay frame as in training: bay heading +x, bay centre at the origin, target base_link 1.45 m behind
the centre. Training has the kerb on the bay's RIGHT and the road on its LEFT; a real bay may be the
mirror image, so the side is fixed at the start of the manoeuvre from where the car is (it arrives
from the road) and y / yaw / steering are mirrored when needed.
Observation: same as training - pose, speed, steering and 12 sector distances (sectors()) from the
lidar points. Without the model the geometric manoeuvre of the local planner is used.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rl'))
DEFAULT_MODEL = os.path.normpath(os.path.join(HERE, '..', 'rl', 'models', 'park_sac_her_best.zip'))
TARGET = np.array([-1.45, 0.0, 0.0])


class Parker:
    def __init__(self, model_path=DEFAULT_MODEL):
        self.model = None
        self.source = 'geometric'
        self.mirror = None
        if model_path and os.path.exists(model_path):
            try:
                from stable_baselines3 import SAC
                from park_env import ParkEnv, sectors
                self.sectors = sectors
                self.model = SAC.load(model_path, env=ParkEnv(), device='cpu')
                self.source = 'rl'
            except Exception as e:                       # pragma: no cover
                self.error = str(e)

    def start(self):
        self.mirror = None

    def step(self, pose, bay, v, steer, points_base):
        """pose (x, y, yaw) world; bay (x, y, yaw) world bay centre; points_base: lidar Nx>=2 in
        base_link. Returns (steer, speed, done)."""
        x, y, yaw = pose
        bx, by, byaw = bay
        c, s = math.cos(byaw), math.sin(byaw)
        lx, ly = c * (x - bx) + s * (y - by), -s * (x - bx) + c * (y - by)
        lyaw = math.atan2(math.sin(yaw - byaw), math.cos(yaw - byaw))
        if self.mirror is None:
            self.mirror = ly < 0                          # car arrives on the bay's right: mirror
        m = -1.0 if self.mirror else 1.0
        ly, lyaw = m * ly, m * lyaw
        P = np.asarray(points_base, float)[:, :2] if points_base is not None and len(points_base) else np.zeros((0, 2))
        if self.mirror and len(P):
            P = P * np.array([1.0, -1.0])
        o = np.r_[lx / 5, ly / 5, math.cos(lyaw), math.sin(lyaw), v / 1.2, m * steer / 0.45, self.sectors(P)]
        obs = {'observation': o.astype(np.float32),
               'achieved_goal': np.array([lx / 5, ly / 5, math.cos(lyaw), math.sin(lyaw)], np.float32),
               'desired_goal': np.array([TARGET[0] / 5, 0.0, 1.0, 0.0], np.float32)}
        done = math.hypot(lx - TARGET[0], ly) < 0.25 and abs(lyaw) < math.radians(5) and abs(v) < 0.3
        if done:
            return 0.0, 0.0, True
        a = self.model.predict(obs, deterministic=True)[0]
        return m * float(np.clip(a[0], -1, 1)) * 0.45, float(np.clip(a[1], -1, 1)) * 1.2, False
