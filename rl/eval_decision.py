#!/usr/bin/env python3
"""Evaluate give-way decision policies in rl/decision_env.py.

  python3 rl/eval_decision.py [--policy rl/models/giveway_ppo.zip] [--episodes 300]

Compares the hand-written rule (4 s gap) with the RL policy (+ safety envelope): cleared %,
collisions, forced braking of priority vehicles, illegal entries, mean time to clear.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from decision_env import GiveWayEnv, rule_policy, safety_envelope  # noqa: E402


def run(policy, episodes, seed=1000, rates=(0.0, 0.35)):
    env = GiveWayEnv(seed=seed, rate_range=rates)
    res = {'cleared': 0, 'collision': 0, 'timeout': 0, 'forced': 0, 'bad': 0, 'times': []}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        forced = False
        while True:
            a = policy(env, obs)
            obs, r, term, trunc, info = env.step(a)
            forced |= bool(info.get('forced_brake'))
            if term or trunc:
                break
        k = info.get('result', 'timeout')
        res[k] = res.get(k, 0) + 1
        res['forced'] += forced
        res['bad'] += bool(info.get('bad_entry'))
        if k == 'cleared':
            res['times'].append(info['time'])
    n = episodes
    t = np.array(res['times']) if res['times'] else np.zeros(1)
    return (f"cleared {100 * res['cleared'] / n:5.1f}%  collisions {res['collision']:3d}  forced-brake {res['forced']:3d}  "
            f"illegal-entry {res['bad']:3d}  timeouts {res['timeout']:3d}  time to clear mean {t.mean():5.1f} s "
            f"p90 {np.percentile(t, 90):5.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default=None)
    ap.add_argument('--episodes', type=int, default=300)
    a = ap.parse_args()
    print('rule (4 s)       ', run(lambda env, o: rule_policy(env), a.episodes))
    print('rule (3 s)       ', run(lambda env, o: rule_policy(env, 3.0), a.episodes))
    print('safe-gap greedy  ', run(lambda env, o: 2, a.episodes))
    if a.policy:
        from stable_baselines3 import PPO
        m = PPO.load(a.policy, device='cpu')
        print('RL + envelope    ', run(lambda env, o: safety_envelope(env, int(m.predict(o, deterministic=True)[0])),
                                       a.episodes))
        print('RL (no envelope) ', run(lambda env, o: int(m.predict(o, deterministic=True)[0]), a.episodes))


if __name__ == '__main__':
    main()
