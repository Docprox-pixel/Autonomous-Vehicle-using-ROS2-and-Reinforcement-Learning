#!/usr/bin/env python3
"""Train / evaluate the RL STEERING residual (rl/steer_env.py, real routes of the lane map).

  python3 rl/train_steer.py --steps 600000          -> rl/models/steer_sac.zip
  python3 rl/train_steer.py --eval rl/models/steer_sac.zip
Compared with the planner alone (residual 0): mean / p95 / worst lateral error, zig-zag (mean
|change of the steering command| per step), routes left (> 1.5 m off).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from steer_env import SteerEnv  # noqa: E402


def evaluate(policy, episodes=150, seed=9000):
    env = SteerEnv(seed=seed)
    lats, dcmd, off, worst = [], [], 0, []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        w, prev = 0.0, env.cmd
        while True:
            o, r, te, tr, info = env.step(policy(o))
            lats.append(info['lat'])
            w = max(w, info['lat'])
            dcmd.append(abs(env.cmd - prev))
            prev = env.cmd
            if te or tr:
                off += info.get('result') == 'off'
                break
        worst.append(w)
    return (f'mean |lat| {np.mean(lats):.3f} m  p95 {np.percentile(lats, 95):.2f} m  worst-per-route median '
            f'{np.median(worst):.2f} / max {np.max(worst):.2f} m  zig-zag {np.mean(dcmd) * 1000:.1f} mrad/step  off {off}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=600000)
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'steer_sac.zip'))
    ap.add_argument('--eval', default=None)
    a = ap.parse_args()
    if a.eval:
        from stable_baselines3 import SAC
        m = SAC.load(a.eval, device='cpu')
        print('planner only ', evaluate(lambda o: np.zeros(1, np.float32)))
        print('planner + RL ', evaluate(lambda o: m.predict(o, deterministic=True)[0]))
        return
    import torch
    torch.set_num_threads(4)
    from stable_baselines3 import SAC
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = make_vec_env(lambda: SteerEnv(), n_envs=8, seed=21, vec_env_cls=SubprocVecEnv)
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=400000, batch_size=512, gamma=0.98,
                train_freq=8, gradient_steps=4, policy_kwargs={'net_arch': [128, 128]}, device='cuda', verbose=0)
    chunk = 50000
    for k in range(0, a.steps, chunk):
        model.learn(chunk, reset_num_timesteps=False)
        model.save(a.out)
        print(f'{k + chunk} steps saved -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
