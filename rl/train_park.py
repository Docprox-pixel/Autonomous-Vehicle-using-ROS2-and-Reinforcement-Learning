#!/usr/bin/env python3
"""Train / evaluate the RL PARKING policy: goal-conditioned SAC + Hindsight Experience Replay.

  python3 rl/train_park.py --steps 1500000          -> rl/models/park_sac_her.zip
  python3 rl/train_park.py --eval rl/models/park_sac_her.zip
Evaluation: success (in the bay within 0.25 m / 5 deg), collisions, time, gear changes.
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from park_env import ParkEnv  # noqa: E402


def evaluate(model, episodes=300, seed=4000):
    env = ParkEnv(seed=seed)
    ok, coll, times, gears, final = 0, 0, [], [], []
    for ep in range(episodes):
        o, _ = env.reset(seed=seed + ep)
        g, prev = 0, 0.0
        while True:
            a = model.predict(o, deterministic=True)[0]
            o, r, te, tr, info = env.step(a)
            if env.v * prev < 0:
                g += 1
            prev = env.v if abs(env.v) > 0.05 else prev
            if te or tr:
                break
        ok += info['is_success']
        coll += info['collision']
        final.append(np.hypot(env.x - env.goal[0], env.y - env.goal[1]))
        if info['is_success']:
            times.append(env.t * 0.1)
            gears.append(g)
    return (f'success {100 * ok / episodes:5.1f}%  collisions {coll}  time {np.mean(times) if times else 0:.1f} s  '
            f'gear changes {np.mean(gears) if gears else 0:.1f}  final distance median {np.median(final):.2f} m')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500000)
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'park_sac_her.zip'))
    ap.add_argument('--eval', default=None)
    a = ap.parse_args()
    from stable_baselines3 import SAC, HerReplayBuffer
    if a.eval:
        m = SAC.load(a.eval, env=ParkEnv(), device='cpu')
        print('RL parking ', evaluate(m))
        return
    import torch
    torch.set_num_threads(4)
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    # 8 environments: every gradient step sees 8 new transitions (1 env: 50k steps took 10 min)
    env = make_vec_env(lambda: ParkEnv(), n_envs=8, seed=3, vec_env_cls=SubprocVecEnv)
    model = SAC('MultiInputPolicy', env, replay_buffer_class=HerReplayBuffer,
                replay_buffer_kwargs=dict(n_sampled_goal=4, goal_selection_strategy='future'),
                learning_rate=1e-3, buffer_size=1000000, batch_size=1024, gamma=0.95, tau=0.05,
                learning_starts=4000, train_freq=1, gradient_steps=2,
                policy_kwargs=dict(net_arch=[256, 256, 256]), device='cuda', verbose=0)
    # reverse curriculum: raise the difficulty whenever the recent success rate passes 60 %
    diff = 0.0
    env.set_attr('difficulty', diff)
    chunk = 20000
    for k in range(0, a.steps, chunk):
        model.learn(chunk, reset_num_timesteps=False)
        res = [x for r in env.get_attr('results') for x in r]
        rate = float(np.mean(res)) if res else 0.0
        if rate > 0.6 and diff < 1.0:
            diff = min(1.0, diff + 0.1)
            env.set_attr('difficulty', diff)
            env.set_attr('results', [])
        model.save(a.out)
        print(f'{k + chunk} steps: success {100 * rate:.0f}% at difficulty {diff:.1f} -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
