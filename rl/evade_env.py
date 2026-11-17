"""RL SUDDEN-OBSTACLE environment: brake, swerve within the free road, or both.

Two-lane road (right-hand traffic). Our lane centre y = 0 (lane -1.75 .. 1.75), the oncoming lane
1.75 .. 5.25. Something appears in our lane 8-40 m ahead: a pedestrian stepping out (0.6 m wide,
may keep walking across at ~1.3 m/s or stop), a stopped car (1.9 m), or an object partly in the
lane (0.5-1.5 m). Oncoming cars come in the other lane in about half the episodes.

Action (2): acceleration [-1, 1] -> [-6, +1] m/s^2 and lateral target [-1, 1] -> [-0.6, +3.5] m.
Hard envelope (also on the car, auto_car_core/evade_rl.py):
  * the lateral target never leaves the road (car centre -0.6 .. +3.5),
  * no target over the centre line (> 0.75) while an oncoming car would meet us within 4 s,
  * the SHIELD brakes at -6 m/s^2 when the swept path hits something within 1.2 s.
Lateral motion: the car moves towards the target at most 2.5 m/s, lateral acceleration <= 4 m/s^2.
Reward: collision -100 (end); shield -3/step; progress 0.05/m; -0.15|y| (stay in the lane),
-0.3/step over the centre line; comfort -0.01 a^2; passed the obstacle by 25 m and back in the
lane (|y| < 0.4) +15 (end); 30 s timeout.
"""
import math

import gymnasium as gym
import numpy as np

DT = 0.1
L_FRONT, L_BACK, HALF_W = 4.05, 1.15, 1.0
Y_MIN, Y_MAX, CENTRE = -0.6, 3.5, 0.75          # car-centre limits; over the centre line beyond 0.75
A_MIN, A_MAX = -6.0, 1.0


class EvadeEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-3, 3, (12,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.x, self.y, self.vy = 0.0, 0.0, 0.0
        self.v = r.uniform(5.0, 13.9)
        self.v0 = self.v
        self.a = 0.0
        self.t = 0.0
        kind = r.choice(['ped', 'ped_walk', 'car', 'object'], p=[.25, .25, .25, .25])
        d = r.uniform(8.0, 40.0)
        if kind.startswith('ped'):
            w, oy = 0.6, r.uniform(-1.5, 0.5)
            vy = r.choice([1.3, -1.3]) if kind == 'ped_walk' else 0.0
        elif kind == 'car':
            w, oy, vy = 1.9, r.uniform(-0.6, 0.3), 0.0
        else:
            w = r.uniform(0.5, 1.5)
            oy = r.choice([-1.75 + w / 2, r.uniform(-1.0, 1.0)])
            vy = 0.0
        self.obs_ = {'x': L_FRONT + d, 'y': oy, 'w': w, 'len': 0.6 if kind.startswith('ped') else
                     (4.5 if kind == 'car' else w), 'vy': vy, 'kind': kind}
        self.oncoming = []
        if r.random() < 0.55:
            x = r.uniform(30, 160)
            for _ in range(int(r.integers(1, 4))):
                self.oncoming.append({'x': x, 'v': r.uniform(6, 13.9)})
                x += r.uniform(25, 80)
        self.shield_steps = 0
        return self._obs(), {}

    # --------------------------------------------------------------- features
    def oncoming_meet(self):
        """Time until the nearest oncoming car meets our car (relative closing)."""
        best = math.inf
        for c in self.oncoming:
            gap = c['x'] - (self.x + L_FRONT)
            if gap > -4.5:
                best = min(best, max(gap, 0.0) / max(self.v + c['v'], 0.5))
        return best

    def _obs(self):
        o = self.obs_
        dx = o['x'] - o['len'] / 2 - (self.x + L_FRONT)
        meet = self.oncoming_meet()
        near = min((c['x'] - self.x for c in self.oncoming if c['x'] - self.x > -5), default=200.0)
        return np.array([self.v / 10, self.a / 6, self.y / 2, self.vy / 2, np.clip(dx, -10, 60) / 30,
                         (o['y'] - o['w'] / 2) / 2, (o['y'] + o['w'] / 2) / 2, o['vy'] / 2,
                         min(meet, 10) / 10, min(near, 200) / 100,
                         1.0 if o['kind'].startswith('ped') else 0.0, (self.y - CENTRE) / 2], np.float32)

    def envelope(self, y_t):
        y_t = float(np.clip(y_t, Y_MIN, Y_MAX))
        if y_t > CENTRE and self.oncoming_meet() < 4.0 and self.y <= CENTRE + 0.05:
            y_t = CENTRE
        return y_t

    def _overlap(self, x, y):
        o = self.obs_
        return (x + L_FRONT > o['x'] - o['len'] / 2 and x - L_BACK < o['x'] + o['len'] / 2 and
                abs(y - o['y']) < HALF_W + o['w'] / 2)

    # --------------------------------------------------------------- dynamics
    def step(self, action):
        a_des = A_MIN + (float(np.clip(action[0], -1, 1)) + 1) * 0.5 * (A_MAX - A_MIN)
        y_t = Y_MIN + (float(np.clip(action[1], -1, 1)) + 1) * 0.5 * (Y_MAX - Y_MIN)
        return self._advance(a_des, y_t)

    def _advance(self, a_des, y_t):
        y_t = self.envelope(y_t)
        # shield: would the current motion hit the obstacle within 1.2 s?
        shield = False
        for k in range(1, 13):
            tt = k * DT
            if self._overlap(self.x + self.v * tt, self.y + self.vy * tt):
                shield = True
                break
        a = A_MIN if shield else float(np.clip(a_des, A_MIN, A_MAX))
        self.a = a
        self.v = max(0.0, self.v + a * DT)
        vy_des = float(np.clip((y_t - self.y) / 0.6, -2.5, 2.5))
        self.vy += float(np.clip(vy_des - self.vy, -4.0 * DT, 4.0 * DT))
        if self.v < 0.5:
            self.vy *= 0.5                                   # a stopped car does not slide sideways
        x0 = self.x
        self.x += self.v * DT
        self.y = float(np.clip(self.y + self.vy * DT, Y_MIN - 0.05, Y_MAX + 0.05))
        self.t += DT
        o = self.obs_
        o['y'] += o['vy'] * DT
        if abs(o['y']) > 6.0:
            o['vy'] = 0.0                                    # walked off the road
        for c in self.oncoming:
            c['x'] -= c['v'] * DT
            # oncoming cars brake for us in their lane, but late (reaction 1 s)
            if self.y > CENTRE and abs(c['x'] - self.x) < c['v'] * 1.5 + 8:
                c['v'] = max(0.0, c['v'] - 4.0 * DT)
        reward = 0.05 * (self.x - x0) - 0.15 * abs(self.y) - 0.01 * a * a
        if self.y > CENTRE:
            reward -= 0.3
        info = {}
        if shield:
            reward -= 3.0
            self.shield_steps += 1
            info['shield'] = True
        if self._overlap(self.x, self.y):
            return self._obs(), reward - 100.0, True, False, {**info, 'result': 'collision'}
        for c in self.oncoming:
            if self.y + HALF_W > 1.75 + 0.1 and abs(c['x'] - (self.x + 1.45)) < 2.6 + 2.3 and \
                    abs(self.y - 3.5) < HALF_W + 1.0:
                return self._obs(), reward - 100.0, True, False, {**info, 'result': 'head_on'}
        if self.x - L_BACK > o['x'] + o['len'] / 2 + 25.0 and abs(self.y) < 0.4:
            return self._obs(), reward + 15.0, True, False, {**info, 'result': 'passed', 'time': self.t}
        if self.t > 30.0:
            return self._obs(), reward, False, True, {**info, 'result': 'timeout'}
        return self._obs(), reward, False, False, info


def rule_action(env):
    """Today's behaviour: the planner's in-lane nudge (<= 0.7 m away from the object) and a
    comfortable stop before it (1.5-3 m/s^2) - never leaving its lane."""
    o = env.obs_
    dx = o['x'] - o['len'] / 2 - (env.x + L_FRONT)
    # nudge away from the object, at most 0.7 m, keeping 0.75 m clearance if possible
    if o['y'] >= env.y:
        y_t = max(-0.6, min(0.0, o['y'] - o['w'] / 2 - HALF_W - 0.75))
    else:
        y_t = min(0.7, max(0.0, o['y'] + o['w'] / 2 + HALF_W + 0.75))
    clear = abs(y_t - o['y']) > HALF_W + o['w'] / 2 + 0.3
    if dx > 0 and not clear:
        room = dx - 2.0
        v_t = 0.0 if room <= 0 else math.sqrt(2 * 3.0 * room)
        a = float(np.clip((v_t - env.v) / DT, -3.0, 1.0))
    else:
        a = float(np.clip((env.v0 - env.v) / DT, -1.5, 1.0))
    if dx < -o['len'] - L_BACK - 2:
        y_t = 0.0
    return a, y_t
