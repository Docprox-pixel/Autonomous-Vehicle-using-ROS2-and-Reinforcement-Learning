#!/usr/bin/env python3
"""Evaluate a driving policy in the kinematic env: the hand-written planner or a trained SAC model.

  python3 rl/evaluate.py --policy planner --episodes 50
  python3 rl/evaluate.py --policy rl/models/sac_driver.zip --episodes 50
Same seeds for every policy, so the comparison is on identical routes and obstacles.
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kin_env import KinematicDrivingEnv, planner_action  # noqa: E402


def run(policy, episodes, seed0=1000, residual=False):
    env = KinematicDrivingEnv()
    if residual:
        from residual import ResidualEnv
        env = ResidualEnv(env)
    model = None
    if policy != 'planner':
        from stable_baselines3 import SAC
        model = SAC.load(policy, device='cpu')
    stats = {'reached': 0, 'collision': 0, 'off_route': 0, 'lat': [], 'time': [], 'interventions': 0, 'steps': 0, 'jerk': []}
    t0 = time.time()
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed0 + ep)
        prev, done, steps, inter, prev_a = 0.0, False, 0, 0, None
        while not done:
            if model is None:
                a = planner_action(env, prev)
                prev = a[0] * 0.45
            else:
                a, _ = model.predict(obs, deterministic=True)
            obs, r, term, trunc, info = env.step(a)
            if residual:
                a = info['final_action']
            done = term or trunc
            steps += 1
            inter += info['intervened']
            stats['lat'].append(info['lat'])
            if prev_a is not None:
                stats['jerk'].append(abs(a[0] - prev_a[0]))
            prev_a = a
        stats['reached'] += info['reached']
        stats['collision'] += info['collision']
        stats['off_route'] += info.get('off_route', False)
        stats['interventions'] += inter
        stats['steps'] += steps
        if info['reached']:
            stats['time'].append(steps * 0.1 / max(info['route_len'], 1) * 100)   # s per 100 m
    n = episodes
    print(f"{policy}: {n} episodes in {time.time() - t0:.0f}s | success {100 * stats['reached'] / n:.0f}% | "
          f"collisions {stats['collision']} | off-route {stats['off_route']} | mean lane error {np.mean(stats['lat']):.2f} m | "
          f"time {np.mean(stats['time']) if stats['time'] else float('nan'):.1f} s/100 m | "
          f"shield interventions {100 * stats['interventions'] / max(stats['steps'], 1):.1f}% of steps | "
          f"steering change {np.mean(stats['jerk']):.3f}/step")
    return stats


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', default='planner')
    ap.add_argument('--episodes', type=int, default=50)
    ap.add_argument('--residual', action='store_true', help='policy is a residual on top of the planner')
    a = ap.parse_args()
    run(a.policy, a.episodes, residual=a.residual)
