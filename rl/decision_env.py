"""Decision-making environment: WHEN to go at a give-way point (junction / roundabout entry).

The hand-written planner keeps driving (lanes, turns, speed in curves); the RL policy only makes
the decision a human driver makes at the line: YIELD (stop / stay stopped at the line), CREEP (edge
up to the line at walking pace, ready to go), or GO (enter and clear the conflict zone).

Scene (ego frame): our car drives along +x towards the give-way line at x = 0. Two crossing traffic
lanes pass in front of it: lane 0 (from the LEFT, moving -y, nearer, x = 1.75 + 1.0) and lane 1
(from the RIGHT, moving +y, x = 5.25 + 1.0). Arrivals are Poisson with a random rate per episode
(empty road ... 0.35 veh/s per lane, ~1260 veh/h), speeds 5-13 m/s, some vehicles slower; crossing vehicles brake
(max 4 m/s^2) only if our car is already in front of them (they have priority, they do not yield).

Observation (21, all ~[-1, 1]): ego distance to the line, speed, time waited; for each lane the 3
nearest vehicles still approaching or in the conflict zone: time to reach the conflict point,
distance, speed (padded with 'none').
Actions: 0 YIELD, 1 CREEP, 2 GO. Decision every 0.2 s; ego longitudinal motion from the action.
Once the front bumper is past the line the car is committed (GO): it never stops inside the junction.
Reward: progress (+0.05/m), waiting (-0.02/step while stopped with a gap that a reasonable driver
would take is NOT penalised separately - the policy learns the trade-off), collision -100 (end),
forcing a priority vehicle to brake hard (> 2.5 m/s^2) -25, entering with a vehicle < 1.5 s from
the conflict point (illegal) -5, clearing the junction +20 (end), timeout 60 s -10.
"""
import math

import gymnasium as gym
import numpy as np

DT = 0.1
DECIDE = 2                 # decision every 0.2 s
CAR_FRONT, CAR_BACK, HALF_W = 4.05, 1.15, 1.0
LANES_X = (2.75, 6.25)     # crossing lane centres (x) beyond the give-way line
LANE_DIR = (-1.0, 1.0)     # lane 0 from the left (moving -y), lane 1 from the right (+y)
OTHER_L, OTHER_W = 4.6, 1.9
V_GO, V_CREEP, A_GO, A_BRAKE = 7.0, 1.5, 2.0, 3.5
SAFE_TTC = 1.5             # hard envelope: never enter with a vehicle closer than this (s)


def t_clear(dist, v, a=A_GO, vmax=V_GO):
    """Time to drive `dist` m from speed v, accelerating at a up to vmax."""
    if dist <= 0:
        return 0.0
    t_acc = max(vmax - v, 0.0) / a
    d_acc = v * t_acc + 0.5 * a * t_acc * t_acc
    if d_acc >= dist:
        return (-v + math.sqrt(v * v + 2 * a * dist)) / a
    return t_acc + (dist - d_acc) / vmax


def lane_clear_dist(bumper_to_line, lane):
    """Distance our FRONT bumper must travel until the REAR has cleared crossing lane `lane`."""
    return bumper_to_line + LANES_X[lane] + OTHER_W / 2 + CAR_FRONT + CAR_BACK


class Vehicle:
    __slots__ = ('lane', 'y', 'v', 'v0', 'braked')

    def __init__(self, lane, y, v):
        self.lane, self.y, self.v, self.v0, self.braked = lane, y, v, v, 0.0


class GiveWayEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, seed=None, rate_range=(0.0, 0.35)):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.rate_range = rate_range
        self.observation_space = gym.spaces.Box(-1.5, 1.5, (21,), np.float32)
        self.action_space = gym.spaces.Discrete(3)

    # ------------------------------------------------------------------ scene
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.x = -self.rng.uniform(18.0, 35.0)        # ego FRONT bumper position (line at 0)
        self.v = self.rng.uniform(3.0, 8.0)
        self.t = 0.0
        self.waited = 0.0
        self.rates = [self.rng.uniform(*self.rate_range) for _ in range(2)]
        self.cars = []
        for lane in (0, 1):                            # pre-populate the streams
            y = -LANE_DIR[lane] * 120.0
            while True:
                y += LANE_DIR[lane] * self._gap(lane)
                if LANE_DIR[lane] * y > 40.0:
                    break
                self.cars.append(Vehicle(lane, y, self._speed()))
        self.spawn = [self._gap(l) / 9.0 for l in (0, 1)]
        self.entered_bad = False
        self.last_action = 0
        return self._obs(), {}

    def _speed(self):
        return float(self.rng.uniform(5.0, 13.0) if self.rng.random() > 0.15 else self.rng.uniform(2.5, 5.0))

    def _gap(self, lane):
        r = self.rates[lane]
        return 1e9 if r < 0.02 else float(max(OTHER_L + 3.0, self.rng.exponential(1.0 / r) * 9.0))

    # ------------------------------------------------------------- features
    def _arrivals(self, lane):
        """(time to conflict, signed distance to conflict, speed) of the approaching vehicles."""
        out = []
        for c in self.cars:
            if c.lane != lane:
                continue
            d = -LANE_DIR[lane] * c.y                 # > 0: still approaching the crossing (y = 0)
            if d < -(OTHER_L + HALF_W + 1.0):
                continue                               # gone past
            ttc = max(d - (HALF_W + OTHER_L / 2), 0.0) / max(c.v, 0.1)
            out.append((ttc, d, c.v))
        out.sort(key=lambda q: q[1])
        return out

    def _obs(self):
        o = [np.clip(self.x / 30.0, -1.5, 1.5), self.v / 10.0, min(self.waited / 20.0, 1.5)]
        for lane in (0, 1):
            arr = [q for q in self._arrivals(lane)][:3]
            for k in range(3):
                if k < len(arr):
                    ttc, d, v = arr[k]
                    o += [min(ttc, 10.0) / 10.0, np.clip(d / 50.0, -1.0, 1.0), v / 15.0]
                else:
                    o += [1.0, 1.0, 0.0]
        return np.array(o, np.float32)

    def gap_ok(self, margin=0.5):
        """Envelope: for every crossing lane the next vehicle arrives after we have cleared it."""
        for lane in (0, 1):
            need = t_clear(lane_clear_dist(-self.x, lane), self.v) + margin
            for ttc, d, v in self._arrivals(lane):
                if d > -(OTHER_L / 2 + HALF_W) and ttc < need:
                    return False
        return True

    def min_ttc(self):
        """Time until the next priority vehicle reaches the conflict zone (lanes in our path)."""
        ttc = math.inf
        for lane in (0, 1):
            for t_, d, v in self._arrivals(lane):
                if d > -(OTHER_L / 2 + HALF_W):
                    ttc = min(ttc, t_)
        return ttc

    # ------------------------------------------------------------- dynamics
    def step(self, action):
        reward, done, info = 0.0, False, {}
        stop_dist = self.v * self.v / (2 * A_BRAKE)
        if self.x > 0.5 or (self.last_action == 2 and self.x + stop_dist > -0.3):
            action = 2                 # COMMITTED: past the line, or a GO that can no longer stop before it
        elif action == 2 and not self.gap_ok():   # (x <= 0.5: still at the line)
            action = 1 if self.x < -3.0 else 0       # envelope: not a safe gap -> creep / hold
            reward -= 0.05
        self.last_action = action
        if self.x < 0.3 and self.x + self.v * 0.4 >= -0.2 and action == 2 and self.min_ttc() < SAFE_TTC:
            self.entered_bad = True
            reward -= 5.0
        for _ in range(DECIDE):
            # ego longitudinal control from the decision
            if action == 2:
                target, a_max = V_GO, A_GO
            elif action == 1:
                # creep: walking pace, but slowing in time to stop with the bumper AT the line
                room = max(-0.3 - self.x, 0.0)
                target, a_max = min(V_CREEP, math.sqrt(2.0 * 1.5 * room)), A_GO
                if room < 0.05:
                    target = 0.0
            else:
                target, a_max = 0.0, A_GO
            stop_at = -0.3                                # stop with the bumper at the line
            if target == 0.0 and self.x < stop_at:
                dist = stop_at - self.x
                need = self.v * self.v / max(2 * dist, 0.05)
                acc = -min(max(need, 0.0), 6.0) if need > 0.5 else -min(self.v / DT, A_BRAKE)
                if dist < 0.05:
                    acc = -self.v / DT
            elif target == 0.0:
                acc = -min(self.v / DT, A_BRAKE)        # past the line already: brake (bad idea)
            else:
                acc = float(np.clip((target - self.v) / 1.0, -A_BRAKE, a_max))
            x0 = self.x
            self.v = max(0.0, self.v + acc * DT)
            self.x += self.v * DT
            reward += 0.05 * (self.x - x0)
            if self.v < 0.2:
                self.waited += DT
                reward -= 0.02
            # crossing traffic: keeps its speed, brakes only for our car in front of it
            for c in self.cars:
                ahead = self._ego_in_lane(c.lane)
                d = -LANE_DIR[c.lane] * c.y
                gap = d - (HALF_W + OTHER_L / 2)
                if ahead and 0.0 < gap < c.v * c.v / 2.0 / 2.0 + 4.0:
                    dec = min(c.v * c.v / max(2 * gap, 0.3), 8.0)
                    if dec > 2.5:
                        c.braked = max(c.braked, dec)
                    c.v = max(0.0, c.v - min(dec, 8.0) * DT)
                else:
                    c.v = min(c.v0, c.v + 1.5 * DT)
                c.y += LANE_DIR[c.lane] * c.v * DT
            # spawn new arrivals far upstream
            for lane in (0, 1):
                self.spawn[lane] -= DT
                if self.spawn[lane] <= 0:
                    self.cars.append(Vehicle(lane, -LANE_DIR[lane] * 120.0, self._speed()))
                    self.spawn[lane] = self._gap(lane) / 9.0
            self.cars = [c for c in self.cars if LANE_DIR[c.lane] * c.y < 60.0]
            self.t += DT
            if self._collision():
                return self._obs(), reward - 100.0, True, False, {'result': 'collision'}
        for c in self.cars:
            if c.braked > 2.5:
                reward -= 25.0
                info['forced_brake'] = True
                c.braked = -1e9                               # count once per vehicle
        if self.x - CAR_BACK > LANES_X[1] + OTHER_W / 2 + 1.0:
            return self._obs(), reward + 20.0, True, False, {'result': 'cleared', 'time': self.t,
                                                             'bad_entry': self.entered_bad, **info}
        if self.t > 60.0:
            return self._obs(), reward - 10.0, False, True, {'result': 'timeout', **info}
        return self._obs(), reward, done, False, info

    def _ego_in_lane(self, lane):
        lx = LANES_X[lane]
        return self.x > lx - OTHER_W / 2 - 0.3 and self.x - CAR_FRONT - CAR_BACK < lx + OTHER_W / 2

    def _collision(self):
        for c in self.cars:
            lx = LANES_X[c.lane]
            if (self.x > lx - OTHER_W / 2 and self.x - CAR_FRONT - CAR_BACK < lx + OTHER_W / 2 and
                    abs(c.y) < OTHER_L / 2 + HALF_W):
                return True
        return False


def rule_policy(env, horizon=4.0):
    """The current hand-written rule (traffic_rules.py): go only if no priority vehicle reaches the
    junction within `horizon` s, else stop at the line."""
    return 2 if env.min_ttc() > horizon else 0


def safety_envelope(env, action):
    """Hard rule over any policy (also applied inside step()): GO only into a gap we can clear."""
    if action == 2 and env.x <= 0.0 and not env.gap_ok():
        return 0
    return action
