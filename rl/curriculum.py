#!/usr/bin/env python3
"""Speed curriculum with a STRICT reward: learn to drive at 50 km/h first, then 60, 80, 100 km/h.

  python3 rl/curriculum.py                       # all stages, village map incl. the ring motorway
  python3 rl/curriculum.py --stages 13.9 16.7    # only the first two

Why a curriculum: a policy that is not yet stable at 50 km/h only gets worse at 100 km/h (larger
stopping distances, smaller margins in curves). Each stage must pass an EXAM before the speed cap
is raised: >= 90 % of 30 fixed test routes completed, with ZERO collisions and ZERO lane departures.
If a stage fails its exam after the extra training chunks, the curriculum stops there (and says
so) - it never moves on to a higher speed with an unstable driver.

The policy is RESIDUAL on top of the rule-based planner (rl/residual.py): the planner already obeys
the rules, the policy learns corrections, so even early in training the car is never worse than it.

Strict reward per 0.1 s step:
  + metres of progress along the route
  - 1.0 * max(0, lane-centre error - 0.5 m)^2      (stay centred in the lane)
  - 2.0 * max(0, speed - speed limit of the road)  (per m/s over the limit from the lane map)
  - 5 if the safety shield had to cut the speed     (drive so the shield never has to intervene)
  - 0.5 * |steering change| - 0.03 * |acceleration|  (smooth, stable driving)
  episode ends with -100: collision, lane departure (> 1.8 m from the lane centre; 3.5 m in junctions),
                         or WRONG SIDE (two-way road: car centre over the centre line, oncoming lane)
  + 0.05 per step on a two-way road while within 0.5 m of the lane centre (keep right, stay centred)
  Oncoming traffic: on 70 % of episodes 1-2 cars drive toward the car in the opposite lane.
  episode ends with +100: goal reached;  -20 if stuck (no progress for 20 s)
"""
import argparse
import json
import math
import os
import sys
import time

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
import kin_env  # noqa: E402
from kin_env import DT, MAX_STEER, KinematicDrivingEnv, to_base  # noqa: E402

VILLAGE = os.path.join(HERE, '..', 'worlds', 'village_net', 'sumo.net.xml')
STAGES = [13.9, 16.7, 22.2, 27.8]            # m/s: 50, 60, 80, 100 km/h
RESIDUAL_SCALE = 0.3


class StrictEnv(KinematicDrivingEnv):
    """Kinematic env with a speed cap (curriculum stage), map speed limits and the strict reward."""

    def __init__(self, v_cap=13.9, highway_start=0.3, **kw):
        super().__init__(net_file=kw.pop('net_file', VILLAGE), **kw)
        self.v_cap = v_cap
        self.highway_start = highway_start
        types = self.map.road_types()
        self.hw_lanes = [l for l in self.map.road_lanes if types.get(l.edge) == 'highway' and l.length > 60]
        A, B, S, H, T, Wd = [], [], [], [], [], []
        for lane in self.map.road_lanes:
            for a, b in zip(lane.pts[:-1], lane.pts[1:]):
                A.append(a), B.append(b), S.append(lane.speed), H.append(types.get(lane.edge) == 'highway')
                T.append(types.get(lane.edge) == 'two_way'), Wd.append(lane.width)
        self.segA, self.segB, self.segS, self.segH = np.array(A), np.array(B), np.array(S), np.array(H)
        self.segT, self.segW = np.array(T), np.array(Wd)
        self.oncoming_rate = 0.7          # episodes with oncoming traffic on two-way roads

    # speed limit and road type at the car's position (nearest lane segment of the map)
    def road_here(self):
        p = np.array([self.x, self.y])
        ab = self.segB - self.segA
        t = np.clip(((p - self.segA) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
        d = np.hypot(*(self.segA + t[:, None] * ab - p).T)
        i = int(np.argmin(d))
        self.on_lane = d[i] < 2.0            # over a mapped lane (else: inside a junction)
        self.two_way_here, self.lane_w_here = bool(self.segT[i]), float(self.segW[i])
        return float(self.segS[i]), bool(self.segH[i])

    def reset(self, seed=None, options=None):
        self.oncoming = []                   # cleared before the base reset builds the new route
        obs, info = super().reset(seed=seed, options=options)
        rng = self.np_random
        if self.hw_lanes and rng.random() < self.highway_start:     # start some episodes on the motorway
            for _ in range(30):
                lane = self.hw_lanes[rng.integers(len(self.hw_lanes))]
                s0 = rng.uniform(2.0, max(3.0, lane.length - 60.0))
                p0 = lane.at(s0)
                d = lane.at(s0 + 1.0) - p0
                yaw = math.atan2(d[1], d[0])
                gl = self.hw_lanes[rng.integers(len(self.hw_lanes))]
                route = self.map.route(tuple(p0), yaw, tuple(gl.at(rng.uniform(0.3, 0.7) * gl.length)))
                if route is None:
                    continue
                length = float(np.hypot(*np.diff(route, axis=0).T).sum())
                if 150.0 < length < 700.0:
                    self.route, self.route_len = route, length
                    self.route_s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(route, axis=0).T))]
                    self.x, self.y, self.yaw = float(p0[0]), float(p0[1]), yaw
                    self.boxes = []
                    break
        # junction sections of the route (+/- 8 m): route points not on a road lane centreline are
        # inside a junction (internal lanes / turn fillets). There a car this size must swing wide
        # in 2.5 m village lanes, so the lane-departure limit there is 3.5 m instead of 1.8 m.
        P = self.route
        ab = self.segB - self.segA
        L2 = np.maximum((ab ** 2).sum(1), 1e-9)
        dmin = np.empty(len(P))
        for k0 in range(0, len(P), 64):
            q = P[k0:k0 + 64]
            t = np.clip(((q[:, None, :] - self.segA[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
            dmin[k0:k0 + 64] = np.hypot(*(self.segA[None] + t[..., None] * ab[None] - q[:, None, :]).transpose(2, 0, 1)).min(1)
        inj = dmin > 0.5
        widen = np.zeros(len(P), bool)
        for i in np.flatnonzero(inj):
            widen |= np.abs(self.route_s - self.route_s[i]) < 8.0
        self.route_junction = widen
        # two-way flag per route point and the opposite lane (route shifted one lane width left)
        ab_r = np.gradient(self.route, axis=0)
        ab_r /= np.maximum(np.hypot(*ab_r.T), 1e-9)[:, None]
        nl = np.c_[-ab_r[:, 1], ab_r[:, 0]]
        j = np.empty(len(self.route), int)
        for k0 in range(0, len(self.route), 64):
            q = self.route[k0:k0 + 64]
            t = np.clip(((q[:, None, :] - self.segA[None]) * ab[None]).sum(-1) / L2[None], 0, 1)
            j[k0:k0 + 64] = np.argmin(np.hypot(*(self.segA[None] + t[..., None] * ab[None] - q[:, None, :]).transpose(2, 0, 1)), 1)
        self.route_two_way = self.segT[j] & ~self.route_junction
        self.route_lane_w = self.segW[j]
        self.opp_path = self.route + nl * self.route_lane_w[:, None]
        self.oncoming = []
        if rng.random() < self.oncoming_rate and self.route_two_way.any():
            idx = np.flatnonzero(self.route_two_way & (self.route_s > 30.0))
            for _ in range(int(rng.integers(1, 3))):
                if len(idx):
                    k = int(rng.choice(idx))
                    self.oncoming.append({'s': float(self.route_s[k]), 'v': float(rng.uniform(0.4, 0.9) * self.v_cap)})
        for car in self.leads:                                       # lead cars scale with the stage
            car['v'] = float(rng.uniform(0.3, 0.8) * self.v_cap)
        self.last_progress_t = 0
        self._best_prog = 0.0            # per-episode progress bookkeeping (stuck detection, reward)
        self._last_prog = 0.0
        self.prev_v = 0.0
        return self._obs(), info

    def _oncoming_pose(self, car):
        s_ = car['s']
        p = np.array([np.interp(s_, self.route_s, self.opp_path[:, 0]), np.interp(s_, self.route_s, self.opp_path[:, 1])])
        q = np.array([np.interp(s_ + 1.0, self.route_s, self.opp_path[:, 0]), np.interp(s_ + 1.0, self.route_s, self.opp_path[:, 1])])
        h = math.atan2(q[1] - p[1], q[0] - p[0]) + math.pi          # drives AGAINST the route direction
        return p, h

    def _obstacle_world(self):
        pts = super()._obstacle_world()
        extra = []
        for car in getattr(self, 'oncoming', []):
            c, hd = self._oncoming_pose(car)
            L, W = 4.5, 1.8
            u, w = np.array([math.cos(hd), math.sin(hd)]), np.array([-math.sin(hd), math.cos(hd)])
            gl, gw = np.linspace(-L / 2, L / 2, 18), np.linspace(-W / 2, W / 2, 8)
            extra.append(np.vstack([c + np.outer(gl, u) + side * W / 2 * w for side in (-1, 1)] +
                                   [c + np.outer(gw, w) + side * L / 2 * u for side in (-1, 1)]))
        return np.vstack([pts] + extra) if extra else pts

    def _tracks(self):
        out = super()._tracks()
        for car in getattr(self, 'oncoming', []):
            c, hd = self._oncoming_pose(car)
            cb = to_base(c[None], self.x, self.y, self.yaw)[0]
            rel = hd - self.yaw
            out.append({'pos': (cb[0], cb[1]), 'vel': (car['v'] * math.cos(rel), car['v'] * math.sin(rel)), 'size': (4.5, 1.8)})
        return out

    def step(self, action):
        for car in getattr(self, 'oncoming', []):
            car['s'] -= car['v'] * DT                              # coming toward us
        self.oncoming = [c for c in getattr(self, 'oncoming', []) if c['s'] > -10.0]
        # action -> command with this stage's speed cap
        steer_cmd = float(np.clip(action[0], -1, 1)) * MAX_STEER
        v_cmd = (float(np.clip(action[1], -1, 1)) + 1.0) / 2.0 * self.v_cap
        kin_env.V_MAX, saved = self.v_cap, kin_env.V_MAX
        try:
            a_internal = np.array([steer_cmd / MAX_STEER, v_cmd / self.v_cap * 2.0 - 1.0], np.float32)
            obs, _, term, trunc, info = super().step(a_internal)
        finally:
            kin_env.V_MAX = saved
        limit, hw = self.road_here()
        # lane error of the car BODY centre (1.45 m ahead of the rear axle): the rear axle
        # naturally runs inside the lane in every curve (~0.7 m at R = 7 m), which is not a departure
        cx, cy = self.x + 1.45 * math.cos(self.yaw), self.y + 1.45 * math.sin(self.yaw)
        ab = np.diff(self.route, axis=0)
        t = np.clip(((np.array([cx, cy]) - self.route[:-1]) * ab).sum(1) / np.maximum((ab ** 2).sum(1), 1e-9), 0, 1)
        lat = float(np.hypot(*(self.route[:-1] + t[:, None] * ab - [cx, cy]).T).min())
        info['lat'] = lat
        r = (max(0.0, info['progress'] - getattr(self, '_last_prog', 0.0))
             - 1.0 * max(0.0, lat - 0.5) ** 2
             - 2.0 * max(0.0, self.v - limit)
             - (5.0 if info['intervened'] else 0.0)
             - 0.5 * abs(action[0] - self.prev_action[0]) - 0.03 * abs(self.v - self.prev_v) / DT)
        # (keep_right / wrong_side are computed below; added after)
        self._last_prog = info['progress']
        self.prev_v = self.v
        # lane departure: > 1.8 m from the lane centre on a road; inside a junction (no painted
        # lanes, the turning path may swing wide) the limit is 3.5 m
        i_near = int(np.argmin(np.hypot(self.route[:, 0] - cx, self.route[:, 1] - cy)))
        departure = lat > (3.5 if (self.route_junction[i_near] or not self.on_lane) else 1.8)
        # Two-way road: the body centre must never cross the centre line into the oncoming lane
        tv = self.route[min(i_near + 1, len(self.route) - 1)] - self.route[max(i_near - 1, 0)]
        tv = tv / max(np.hypot(*tv), 1e-9)
        signed = float((np.array([cx, cy]) - self.route[i_near]) @ np.array([-tv[1], tv[0]]))   # + = left
        wrong_side = bool(self.route_two_way[i_near] and signed > self.route_lane_w[i_near] / 2.0)
        keep_right = 0.05 if (self.route_two_way[i_near] and abs(signed) < 0.5) else 0.0
        if info['progress'] > getattr(self, '_best_prog', 0.0) + 0.5:
            self._best_prog, self.last_progress_t = info['progress'], self.t
        stuck = self.t - self.last_progress_t > int(20.0 / DT)
        r += keep_right
        if info['collision']:
            r -= 100.0
        if departure or wrong_side:
            r -= 100.0
        if info['reached']:
            r += 100.0
        if stuck:
            r -= 20.0
        info.update(departure=departure, wrong_side=wrong_side, speeding=self.v > limit + 0.5, limit=limit,
                    highway=hw, stuck=stuck)
        return obs, float(r), bool(term or departure or wrong_side), bool(trunc or stuck), info


class StrictResidualEnv(gym.Wrapper):
    """Residual on top of the planner, with the stage's speed cap passed to the planner too."""

    def __init__(self, env):
        super().__init__(env)
        lo = np.r_[env.observation_space.low, [-1, -1]].astype(np.float32)
        hi = np.r_[env.observation_space.high, [1, 1]].astype(np.float32)
        self.observation_space = gym.spaces.Box(lo, hi, dtype=np.float32)
        self.prev_steer = 0.0

    def base(self):
        from auto_car_core.planner import PlannerParams, plan
        e = self.env.unwrapped
        limit, hw = e.road_here()
        pp = PlannerParams(v_max=min(e.v_cap, limit), v_max_highway=min(e.v_cap, limit))
        rb = to_base(e.route, e.x, e.y, e.yaw)
        st, v, _ = plan(e._last_points, self.prev_steer, pp=pp, path=rb, highway=hw)
        return np.array([st / MAX_STEER, v / e.v_cap * 2.0 - 1.0], np.float32)

    def _aug(self, obs):
        self.b = self.base()
        return np.r_[obs, self.b].astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.prev_steer = 0.0
        return self._aug(obs), info

    def step(self, action):
        a = np.clip(self.b + RESIDUAL_SCALE * np.asarray(action, np.float32), -1.0, 1.0)
        self.prev_steer = float(a[0]) * MAX_STEER
        obs, r, term, trunc, info = self.env.step(a)
        r -= 0.2 * float(np.abs(action).sum())        # stay close to the rule-based driver
        info['final_action'] = a
        return self._aug(obs), r, term, trunc, info


def exam(model, v_cap, episodes=30, seed0=5000):
    env = StrictResidualEnv(StrictEnv(v_cap=v_cap, seed=0))
    s = {'reached': 0, 'collision': 0, 'departure': 0, 'wrong_side': 0, 'speeding_steps': 0, 'steps': 0, 'lat': [], 'vmax': 0.0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=True) if model is not None else (np.zeros(2), None)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            s['steps'] += 1
            s['speeding_steps'] += info['speeding']
            s['lat'].append(info['lat'])
            s['vmax'] = max(s['vmax'], env.env.unwrapped.v)
        s['reached'] += info['reached']
        s['collision'] += info['collision']
        s['departure'] += info['departure']
        s['wrong_side'] += info['wrong_side']
    res = {'v_cap_kmh': round(v_cap * 3.6), 'success': s['reached'] / episodes, 'collisions': s['collision'],
           'lane_departures': s['departure'], 'wrong_side': s['wrong_side'], 'speeding_pct': round(100 * s['speeding_steps'] / max(s['steps'], 1), 2),
           'mean_lane_error_m': round(float(np.mean(s['lat'])), 3), 'top_speed_kmh': round(s['vmax'] * 3.6)}
    res['passed'] = (res['success'] >= 0.9 and res['collisions'] == 0 and res['lane_departures'] == 0
                     and res['wrong_side'] == 0)
    return res


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    ap = argparse.ArgumentParser()
    ap.add_argument('--stages', type=float, nargs='+', default=STAGES)
    ap.add_argument('--steps', type=int, default=40000, help='training steps per chunk')
    ap.add_argument('--extra-chunks', type=int, default=2, help='retries per stage before giving up')
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'curriculum'))
    ap.add_argument('--init', default=None, help='continue from a saved model (.zip)')
    ap.add_argument('--skip-baseline', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, 'curriculum_log.jsonl'), 'a')

    if not a.skip_baseline:
        print('baseline (rule-based planner alone, no RL correction):', flush=True)
    for v in ([] if a.skip_baseline else a.stages):
        r = exam(None, v, episodes=20)
        print('  ', json.dumps(r), flush=True)
        log.write(json.dumps({'stage': 'planner_baseline', **r}) + '\n')
    env = Monitor(StrictResidualEnv(StrictEnv(v_cap=a.stages[0], seed=0)))
    if a.init:
        model = SAC.load(a.init, env=env)
        print(f'continuing from {a.init}', flush=True)
    else:
        model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=200000, batch_size=256, gamma=0.99, tau=0.005,
                    ent_coef='auto_0.05', learning_starts=2000, policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3.0),
                    verbose=0, seed=0)
    for v in a.stages:
        env.env.env.v_cap = v                         # raise the speed cap (Monitor -> wrapper -> env)
        for chunk in range(1 + a.extra_chunks):
            t0 = time.time()
            model.learn(total_timesteps=a.steps, reset_num_timesteps=False, progress_bar=False)
            r = exam(model, v)
            r.update(chunk=chunk, minutes=round((time.time() - t0) / 60, 1))
            print(f'stage {v * 3.6:.0f} km/h, chunk {chunk}: {json.dumps(r)}', flush=True)
            log.write(json.dumps({'stage': v, **r}) + '\n')
            log.flush()
            model.save(os.path.join(a.out, f'stage_{v * 3.6:.0f}kmh'))
            if r['passed']:
                break
        if not r['passed']:
            print(f'stage {v * 3.6:.0f} km/h NOT stable after {1 + a.extra_chunks} chunks: curriculum stops here',
                  flush=True)
            break
    print('done', flush=True)


if __name__ == '__main__':
    main()
