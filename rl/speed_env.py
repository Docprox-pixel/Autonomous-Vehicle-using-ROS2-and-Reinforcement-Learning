"""RL SPEED CONTROL environment: HOW to accelerate / brake towards the planner's target speed.

The planner (scripts/local_planner.py) keeps deciding the TARGET speed - curve limit, speed limit,
2 s headway behind a lead car, comfortable stop before a stop line. Until now a fixed rule moved
the speed towards it (+1.0 / -1.5 / -3.0 m/s^2, jerk <= 1-2 m/s^3). The policy learns that part:
early, gentle braking when something ahead is closing, not crawling when the road is free, firm but
smooth stops at lines, and a timely reaction when something suddenly appears in front.

1-D longitudinal world, 0.1 s steps, episodes of 60 s:
  * a target-speed profile along the road (speed limit 8-13.9 m/s, curves down to 3-6 m/s),
  * a lead vehicle in some episodes: cruising, slow-fast, SUDDEN hard braking (up to 6 m/s^2),
    stopped in the lane, or cutting in a few metres ahead,
  * red lights: a stop line that turns green after a random time,
  * a pedestrian stepping into the lane at a random moment (an obstacle appearing 8-35 m ahead).
The planner's target here is computed like the real one: min(profile, follow speed for a 2 s
headway, sqrt(2 * 1.5 * room) before a stop line / obstacle).

Observation (10): v, a, v_target, v_target - v, free distance ahead (lead / obstacle / stop line,
/60), its rate (closing speed, /10), time-to-collision (/10), stop-line room (/60), speed limit,
whether the free distance is a stop line (1) or an object (0).
Action (1): desired acceleration in [-1, 1] -> [-3, +1] m/s^2 (jerk limited to 2 m/s^3 as before).
Hard limits (also on the car): never above the target speed; the SHIELD emergency-brakes
(-6 m/s^2) when the time-to-collision < 1.2 s or the car cannot stop before an obstacle with
4 m/s^2 - that costs reward, so the policy learns to brake early enough that it never happens.
Reward: progress 0.05/m; -0.3 * (v_target - v)/v_target while slower than needed (no crawling);
comfort -0.02 a^2 - 0.05 jerk^2; shield intervention -3 per step; collision -100 (end);
running a red stop line -50 (end); goal (end of road) +10.
"""
import math

import gymnasium as gym
import numpy as np

DT = 0.1
A_MIN, A_MAX, JERK = -3.0, 1.0, 2.0
SHIELD_DEC, SHIELD_TTC = 6.0, 1.2
CAR_LEN = 4.6


class SpeedEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=None):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-2.0, 2.0, (10,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    # ------------------------------------------------------------------ world
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        r = self.rng
        self.L = 600.0
        # target-speed profile: limit with curves
        self.limit = float(r.choice([8.3, 11.1, 13.9]))
        self.curves = []
        s = r.uniform(30, 80)
        while s < self.L:
            self.curves.append((s, s + r.uniform(10, 40), r.uniform(3.0, 6.5)))
            s += r.uniform(60, 180)
        self.s, self.v, self.a = 0.0, r.uniform(0.0, 6.0), 0.0
        self.t = 0.0
        # lead vehicle
        kind = r.choice(['none', 'cruise', 'slowfast', 'sudden', 'stopped', 'cutin'], p=[.2, .2, .15, .2, .1, .15])
        self.lead = None
        if kind != 'none':
            gap = r.uniform(15, 60) if kind != 'stopped' else r.uniform(60, 200)
            v0 = 0.0 if kind == 'stopped' else r.uniform(3, self.limit)
            self.lead = {'kind': kind, 's': self.s + gap, 'v': v0, 'brake_at': r.uniform(5, 40),
                         'cut_at': r.uniform(5, 30), 'active': kind != 'cutin', 'phase': 0.0}
        # red light
        self.line = None
        if r.random() < 0.5:
            self.line = {'s': r.uniform(80, 400), 'green_at': r.uniform(0, 40)}
        # pedestrian stepping out
        self.ped = None
        if r.random() < 0.35:
            self.ped = {'t': r.uniform(3, 45), 'd': r.uniform(8, 35), 'dur': r.uniform(3, 8), 's': None}
        self.prev_free = None
        self.shield_steps = 0
        return self._obs(), {}

    def profile(self, s):
        v = self.limit
        for a_, b_, vc in self.curves:
            if a_ - 40 < s < b_:                      # the planner previews the curve (braking distance)
                if s < a_:
                    v = min(v, math.sqrt(vc * vc + 2 * 1.5 * (a_ - s)))
                else:
                    v = min(v, vc)
        return v

    def _lead_s(self):
        if self.lead and self.lead['active']:
            return self.lead['s']
        return None

    def _obstacles(self):
        """(distance to the nearest thing ahead from our front bumper, is_line)."""
        cands = []
        ls = self._lead_s()
        if ls is not None:
            cands.append((ls - CAR_LEN - self.s, False, self.lead['v']))
        if self.ped and self.ped['s'] is not None:
            cands.append((self.ped['s'] - self.s, False, 0.0))
        if self.line and self.t < self.line['green_at'] and self.line['s'] > self.s - 0.5:
            cands.append((self.line['s'] - self.s, True, 0.0))
        if not cands:
            return 200.0, False, self.limit
        return min(cands, key=lambda c: c[0])

    def v_target(self):
        """The planner's target speed (as scripts/local_planner.py + planner follow logic)."""
        v = self.profile(self.s)
        free, is_line, v_obj = self._obstacles()
        if free < 200:
            if is_line:
                room = free - 1.0
            else:
                room = free - 2.0 - 2.0 * max(v_obj, 0.0) * 0.5   # stand-off + part of the headway
            follow = v_obj + max(free - 2.0 - 2.0 * v_obj, 0.0) / 2.0 if not is_line else 99
            v = min(v, 0.0 if room <= 0 else math.sqrt(2 * 1.5 * room), follow)
        return max(v, 0.0)

    def _obs(self):
        vt = self.v_target()
        free, is_line, v_obj = self._obstacles()
        rate = 0.0 if self.prev_free is None else (self.prev_free - free) / DT
        ttc = free / rate if rate > 0.1 else 99.0
        room = free if is_line else 200.0
        return np.array([self.v / 10, self.a / 3, vt / 10, (vt - self.v) / 10, min(free, 120) / 60,
                         np.clip(rate / 10, -2, 2), min(ttc, 20) / 10, min(room, 120) / 60, self.limit / 10,
                         1.0 if is_line else 0.0], np.float32)

    # ---------------------------------------------------------------- dynamics
    def step(self, action):
        a_des = float(np.clip(action[0], -1, 1))
        a_des = A_MIN + (a_des + 1) * 0.5 * (A_MAX - A_MIN)
        return self._advance(a_des)

    def _advance(self, a_des):
        vt = self.v_target()
        free, is_line, v_obj = self._obstacles()
        # hard limit: never above the target speed
        if self.v + a_des * DT > vt:
            a_des = min(a_des, (vt - self.v) / DT)
        a = self.a + float(np.clip(a_des - self.a, -JERK * DT * (3 if a_des < self.a else 1), JERK * DT))
        # SHIELD: emergency brake when it would otherwise hit something
        rate = 0.0 if self.prev_free is None else (self.prev_free - free) / DT
        shield = False
        if not is_line and free < 120:
            closing = self.v - v_obj
            need = closing * closing / max(2 * max(free - 1.0, 0.1), 0.1) if closing > 0 else 0.0
            if (closing > 0 and free / closing < SHIELD_TTC) or need > 4.0:
                a, shield = -SHIELD_DEC, True
        jerk = (a - self.a) / DT
        self.a = a
        s0 = self.s
        self.v = max(0.0, self.v + a * DT)
        if self.v == 0.0 and self.a < 0:
            self.a = 0.0
        self.s += self.v * DT
        self.t += DT
        self.prev_free = free
        self._move_world()
        reward = 0.05 * (self.s - s0) - 0.02 * a * a - 0.05 * min(jerk * jerk, 100.0) / 10.0
        vt2 = self.v_target()
        if vt2 > 0.5 and self.v < vt2:
            reward -= 0.3 * (vt2 - self.v) / max(vt2, 1.0)
        info = {}
        if shield:
            reward -= 3.0
            self.shield_steps += 1
            info['shield'] = True
        free2, is_line2, _ = self._obstacles()
        if not is_line2 and free2 < 0.0:
            return self._obs(), reward - 100.0, True, False, {'result': 'collision', **info}
        if self.line and self.t < self.line['green_at'] and s0 <= self.line['s'] < self.s:
            return self._obs(), reward - 50.0, True, False, {'result': 'red_run', **info}
        if self.s >= self.L:
            return self._obs(), reward + 10.0, True, False, {'result': 'goal', **info}
        if self.t >= 60.0:
            return self._obs(), reward, False, True, {'result': 'time', **info}
        return self._obs(), reward, False, False, info

    def _move_world(self):
        r = self.rng
        L = self.lead
        if L:
            if L['kind'] == 'cutin' and not L['active'] and self.t > L['cut_at']:
                L['active'] = True
                L['s'] = self.s + CAR_LEN + r.uniform(5, 14)
                L['v'] = max(self.v - r.uniform(0, 4), 2.0)
            if L['active']:
                lim = self.profile(L['s'])
                if L['kind'] == 'sudden' and self.t > L['brake_at'] and L['phase'] < 1:
                    L['v'] = max(0.0, L['v'] - r.uniform(4, 6) * DT)
                    if L['v'] == 0.0:
                        L['phase'] = 1
                        L['restart'] = self.t + r.uniform(2, 8)
                elif L['kind'] == 'sudden' and L['phase'] >= 1:
                    if self.t > L.get('restart', 1e9):
                        L['v'] = min(L['v'] + 1.5 * DT, lim)
                elif L['kind'] == 'slowfast':
                    want = lim * (0.5 + 0.5 * math.sin(self.t / 6.0) ** 2)
                    L['v'] += float(np.clip(want - L['v'], -2 * DT, 1.2 * DT))
                elif L['kind'] != 'stopped':
                    L['v'] += float(np.clip(min(lim, self.limit * 0.9) - L['v'], -2 * DT, 1.0 * DT))
                L['s'] += L['v'] * DT
                if L['s'] > self.L + 50:
                    L['active'] = False
        P = self.ped
        if P:
            if P['s'] is None and self.t > P['t'] and P['t'] >= 0:
                P['s'] = self.s + CAR_LEN + P['d']
                P['until'] = self.t + P['dur']
            elif P['s'] is not None and self.t > P['until']:
                P['s'], P['t'] = None, -1.0            # crossed: gone


def rule_accel(env):
    """The current hand-written rule in scripts/local_planner.py."""
    vt = env.v_target()
    free, is_line, _ = env._obstacles()
    firm = is_line or vt < env.v - 0.5
    a_min = -3.0 if firm else -1.5
    return float(np.clip((vt - env.v) / DT, a_min, 1.0))
