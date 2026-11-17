#!/usr/bin/env python3
"""Train / evaluate the RL SUDDEN-OBSTACLE policy (rl/evade_env.py).

  python3 rl/train_evade.py --steps 1500000          -> rl/models/evade_sac.zip
  python3 rl/train_evade.py --eval rl/models/evade_sac.zip
Compared with today's behaviour (brake to stop + <= 0.7 m in-lane nudge): collisions, head-on,
emergency-brake episodes, passed (did not have to stop and wait), mean time.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from evade_env import A_MAX, A_MIN, Y_MAX, Y_MIN, EvadeEnv, rule_action  # noqa: E402


def to_action(a, y):
    return np.array([2 * (a - A_MIN) / (A_MAX - A_MIN) - 1, 2 * (y - Y_MIN) / (Y_MAX - Y_MIN) - 1], np.float32)


def evaluate(policy, episodes=400, seed=7000):
    env = EvadeEnv(seed=seed)
    res = {'collision': 0, 'head_on': 0, 'passed': 0, 'timeout': 0}
    shield, times = 0, []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        while True:
            o, r, te, tr, info = env.step(policy(env, o))
            if te or tr:
                break
        res[info.get('result', 'timeout')] += 1
        shield += env.shield_steps > 0
        if info.get('result') == 'passed':
            times.append(info['time'])
    return (f"collisions {res['collision']:3d}  head-on {res['head_on']:3d}  emergency-brake episodes {shield:3d}  "
            f"passed {100 * res['passed'] / episodes:5.1f}%  timeouts {res['timeout']:3d}  "
            f"mean time {np.mean(times) if times else 0:.1f} s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500000)
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'evade_sac.zip'))
    ap.add_argument('--eval', default=None)
    a = ap.parse_args()
    if a.eval is not None:
        print('today (brake+nudge) ', evaluate(lambda env, o: to_action(*rule_action(env))))
        if a.eval:
            from stable_baselines3 import SAC
            m = SAC.load(a.eval, device='cpu')
            print('RL + envelope       ', evaluate(lambda env, o: m.predict(o, deterministic=True)[0]))
        return
    import torch
    torch.set_num_threads(4)
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = make_vec_env(lambda: EvadeEnv(), n_envs=8, seed=31, vec_env_cls=SubprocVecEnv)
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=600000, batch_size=512, gamma=0.99,
                train_freq=8, gradient_steps=4, policy_kwargs={'net_arch': [256, 256]}, device='cuda', verbose=0)
    chunk = 100000
    for k in range(0, a.steps, chunk):
        model.learn(chunk, reset_num_timesteps=False)
        model.save(a.out)
        print(f'{k + chunk} steps saved -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
