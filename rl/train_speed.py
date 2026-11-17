#!/usr/bin/env python3
"""Train / evaluate the RL SPEED controller (rl/speed_env.py).

  python3 rl/train_speed.py --steps 1500000          -> rl/models/speed_sac.zip
  python3 rl/train_speed.py --eval rl/models/speed_sac.zip
Evaluation compares with the current hand-written rule: collisions, red lights run, shield
interventions (emergency braking), mean |jerk|, max braking, mean speed.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from speed_env import A_MAX, A_MIN, SpeedEnv, rule_accel  # noqa: E402


def evaluate(policy, episodes=300, seed=5000):
    env = SpeedEnv(seed=seed)
    out = {'collision': 0, 'red_run': 0, 'shield_eps': 0, 'jerk': [], 'speed': [], 'brake': [], 'shield_steps': 0}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        a_prev = env.a
        while True:
            obs, r, te, tr, info = env.step(policy(env, obs))
            out['jerk'].append(abs(env.a - a_prev) / 0.1)
            a_prev = env.a
            out['speed'].append(env.v)
            out['brake'].append(-env.a)
            if te or tr:
                break
        res = info.get('result')
        if res in ('collision', 'red_run'):
            out[res] += 1
        out['shield_eps'] += env.shield_steps > 0
        out['shield_steps'] += env.shield_steps
    return (f"collisions {out['collision']:3d}  red-run {out['red_run']:3d}  emergency-brake episodes "
            f"{out['shield_eps']:3d} ({out['shield_steps']} steps)  mean|jerk| {np.mean(out['jerk']):.2f}  "
            f"p99 brake {np.percentile(out['brake'], 99):.2f}  mean speed {np.mean(out['speed']):.2f}")


def to_action(a):
    return np.array([2 * (a - A_MIN) / (A_MAX - A_MIN) - 1], np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500000)
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'speed_sac.zip'))
    ap.add_argument('--eval', default=None)
    a = ap.parse_args()
    if a.eval:
        from stable_baselines3 import SAC
        m = SAC.load(a.eval, device='cpu')
        print('rule  ', evaluate(lambda env, o: to_action(rule_accel(env))))
        print('RL    ', evaluate(lambda env, o: m.predict(o, deterministic=True)[0]))
        return
    import torch
    torch.set_num_threads(4)
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = make_vec_env(lambda: SpeedEnv(), n_envs=8, seed=11, vec_env_cls=SubprocVecEnv)
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=500000, batch_size=512, gamma=0.99,
                train_freq=8, gradient_steps=4, policy_kwargs={'net_arch': [128, 128]}, device='cuda', verbose=0)
    chunk = 100000
    for k in range(0, a.steps, chunk):
        model.learn(chunk, reset_num_timesteps=False)
        model.save(a.out)
        print(f'{k + chunk} steps saved -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
