"""RL PARKING environment (goal-conditioned, for SAC + Hindsight Experience Replay).

Like the parking research (goal-conditioned SAC + HER; HOPE adds Reeds-Shepp shortcuts), on OUR
car and OUR bays: parallel bays along the car-park strips (worlds/village_net/parking.yaml, 5.4-5.8 m
x 3.1 m). Bay frame: the bay heading is +x, bay centre at (0, 0); the target is base_link (rear
axle) 1.45 m behind the bay centre with the bay heading (the car centred in the bay) - exactly what
scripts/local_planner.py parking_step aims for.
Scene: parked cars in the bays in front / behind (each present with p 0.75, small offsets), the kerb
at y = -1.9 (the bay's outer side), the road on the other side (car-park wall at y = 8).
Start: alongside the bay, in the road (x -3..12, y 2.4..4.6, heading +-15 deg), as the car arrives.
Training uses a REVERSE CURRICULUM (difficulty 0 -> 1): starts near the parked pose first, moved out
towards the full road start as the success rate passes 60 % (rl/train_park.py).

Action: steering [-1, 1] * 0.45 rad (rate limited 0.15 rad / step), speed [-1, 1] * 1.2 m/s.
observation: x/5, y/5, cos, sin, v/1.2, steer/0.45, 12 sector distances around the car centre
(/6 m, from obstacle outlines - on the car the same function bins the lidar points: sectors()).
achieved / desired goal: (x/5, y/5, cos, sin) of base_link / of the target.
Reward (also used for HER relabelling), in metres: -(sum w |d|)^0.5 with w = (1, 1, 1.5, 1.5),
+2 inside the tolerance; collision -10 (end); success (0.25 m, 5 deg) ends the episode (the car then
stops). 30 s per episode.
"""
import math

import gymnasium as gym
import numpy as np

DT, WB, MAX_STEER = 0.1, 2.995, 0.45
BODY_BACK, BODY_FRONT, HALF_W = 1.15, 4.05, 1.0
TARGET = np.array([-1.45, 0.0, 0.0])
W = np.array([1.0, 1.0, 1.5, 1.5])      # heading counts as much as position (0.1 left the car 6-7 deg crooked)
N_SECT = 12
CAR_CENTRE = 1.45                          # base_link -> body centre


def sectors(pts_car, max_r=6.0):
    """Nearest obstacle distance in 12 sectors around the car CENTRE. pts_car: Nx2 in base_link."""
    d = np.full(N_SECT, max_r)
    if len(pts_car):
        q = pts_car - np.array([CAR_CENTRE, 0.0])
        r = np.hypot(q[:, 0], q[:, 1])
        k = (((np.arctan2(q[:, 1], q[:, 0]) + math.pi) / (2 * math.pi)) * N_SECT).astype(int) % N_SECT
        np.minimum.at(d, k, np.minimum(r, max_r))
    return d / max_r


def rect(cx, cy, yaw, half_l, half_w, step=0.15):
    c, s = math.cos(yaw), math.sin(yaw)
    xs = np.arange(-half_l, half_l + 1e-9, step)
    ys = np.arange(-half_w, half_w + 1e-9, step)
    edge = np.vstack([np.c_[xs, np.full(len(xs), half_w)], np.c_[xs, np.full(len(xs), -half_w)],
                      np.c_[np.full(len(ys), half_l), ys], np.c_[np.full(len(ys), -half_l), ys]])
    return np.c_[cx + c * edge[:, 0] - s * edge[:, 1], cy + s * edge[:, 0] + c * edge[:, 1]]


class ParkEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=None, difficulty=1.0):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.difficulty = difficulty
        self.results = []                      # recent episode outcomes (1 success, 0 not)
        self.observation_space = gym.spaces.Dict({
            'observation': gym.spaces.Box(-3, 3, (6 + N_SECT,), np.float32),
            'achieved_goal': gym.spaces.Box(-5, 5, (4,), np.float32),
            'desired_goal': gym.spaces.Box(-5, 5, (4,), np.float32)})
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.Lb = float(r.choice([5.4, 5.8]))
        obs_pts = [np.c_[np.arange(-20, 20, 0.15), np.full(len(np.arange(-20, 20, 0.15)), -1.9)],   # kerb
                   np.c_[np.arange(-20, 20, 0.3), np.full(len(np.arange(-20, 20, 0.3)), 8.0)]]     # far wall
        self.boxes = []
        for side in (-1, 1):
            if r.random() < 0.75:
                cx = side * self.Lb + r.uniform(-0.3, 0.3)
                cy = r.uniform(-0.25, 0.15)
                yaw = r.uniform(-0.05, 0.05)
                self.boxes.append((cx, cy, yaw, 2.3, 0.95))
                obs_pts.append(rect(cx, cy, yaw, 2.3, 0.95))
        self.obst = np.vstack(obs_pts)
        self.goal = TARGET + np.array([r.uniform(-0.15, 0.15), r.uniform(-0.1, 0.1), 0.0])
        # REVERSE CURRICULUM: difficulty 0 = start almost parked, 1 = arriving alongside on the road
        d = float(self.difficulty)
        for _ in range(50):
            if r.random() < 0.8 * d + 0.2 * (d >= 1.0) or d >= 1.0:
                fx, fy, fyaw = r.uniform(-3.0, 12.0), r.uniform(2.4, 4.6), r.uniform(-0.26, 0.26)
            else:
                fx, fy, fyaw = self.goal[0], self.goal[1], 0.0
            # level 0: only forward / back in the bay (sideways moves need multi-point manoeuvres -
            # the first curriculum started 0.6 m / 8 deg off and never learned anything)
            near = (self.goal[0] + r.uniform(-1.0, 1.0), self.goal[1] + r.uniform(-0.1, 0.1), r.uniform(-0.05, 0.05))
            k = d if d < 1.0 else 1.0
            self.x = near[0] + k * (fx - near[0])
            self.y = near[1] + k * (fy - near[1])
            self.yaw = near[2] + k * (fyaw - near[2])
            if not self._collides(self.x, self.y, self.yaw):
                break
        self.v, self.steer, self.t = 0.0, 0.0, 0
        return self._obs(), {}

    # ------------------------------------------------------------ geometry
    def _body(self, x, y, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        return x + c * CAR_CENTRE, y + s * CAR_CENTRE, (BODY_FRONT + BODY_BACK) / 2, HALF_W, c, s

    def _collides(self, x, y, yaw):
        cx, cy, hl, hw, c, s = self._body(x, y, yaw)
        q = self.obst - np.array([cx, cy])
        lx, ly = c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1]
        return bool(np.any((np.abs(lx) < hl) & (np.abs(ly) < hw)))

    def _goal_vec(self, x, y, yaw):
        return np.array([x / 5, y / 5, math.cos(yaw), math.sin(yaw)], np.float32)

    def _obs(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        q = self.obst - np.array([self.x, self.y])
        pts_car = np.c_[c * q[:, 0] + s * q[:, 1], -s * q[:, 0] + c * q[:, 1]]
        o = np.r_[self.x / 5, self.y / 5, c, s, self.v / 1.2, self.steer / MAX_STEER, sectors(pts_car)]
        return {'observation': o.astype(np.float32),
                'achieved_goal': self._goal_vec(self.x, self.y, self.yaw),
                'desired_goal': self._goal_vec(*self.goal)}

    @staticmethod
    def compute_reward(achieved, desired, info):
        # in METRES (goal vectors hold x/5, y/5): the first version scored 0.25 m and 0 m alike
        a, g = np.asarray(achieved, float), np.asarray(desired, float)
        d = np.abs(a - g) * np.array([5.0, 5.0, 2.0, 2.0])
        r = -np.power(np.dot(d, W), 0.5)
        inside = (np.hypot(d[..., 0], d[..., 1]) < 0.25) & (np.hypot(d[..., 2], d[..., 3]) / 2.0 < math.radians(5))
        r = r + 2.0 * inside
        if isinstance(info, dict):
            return r - (10.0 if info.get('collision') else 0.0)
        coll = np.array([i.get('collision', False) for i in info], float) if len(info) else 0.0
        return r - 10.0 * coll

    def success(self):
        dx, dy = self.x - self.goal[0], self.y - self.goal[1]
        dyaw = math.atan2(math.sin(self.yaw - self.goal[2]), math.cos(self.yaw - self.goal[2]))
        return math.hypot(dx, dy) < 0.25 and abs(dyaw) < math.radians(5)        # (the car then stops)

    def step(self, action):
        st = float(np.clip(action[0], -1, 1)) * MAX_STEER
        self.steer = float(np.clip(st, self.steer - 0.15, self.steer + 0.15))
        v_des = float(np.clip(action[1], -1, 1)) * 1.2
        self.v += float(np.clip(v_des - self.v, -1.5 * DT, 1.5 * DT))
        nx = self.x + self.v * math.cos(self.yaw) * DT
        ny = self.y + self.v * math.sin(self.yaw) * DT
        nyaw = self.yaw + self.v * math.tan(self.steer) / WB * DT
        info = {'collision': False, 'is_success': False}
        if self._collides(nx, ny, nyaw):
            info['collision'] = True
            self.v = 0.0
        else:
            self.x, self.y, self.yaw = nx, ny, nyaw
        self.t += 1
        o = self._obs()
        rew = float(self.compute_reward(o['achieved_goal'], o['desired_goal'], info))
        if info['collision']:
            self.results.append(0)
            return o, rew, True, False, info
        if self.success():
            info['is_success'] = True
            self.results.append(1)
            return o, rew, True, False, info
        if self.t >= 300:
            self.results.append(0)
        self.results = self.results[-100:]
        return o, rew, False, self.t >= 300, info
