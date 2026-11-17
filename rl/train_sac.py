#!/usr/bin/env python3
"""Train the RL driving policy: imitation warm start from the planner, then SAC.

  python3 rl/train_sac.py --steps 300000 --out rl/models/sac_driver

1. Demonstrations: the hand-written planner drives `--demo-steps` steps in the kinematic env.
2. Behaviour cloning: the SAC actor's mean action is trained to match the planner (MSE).
   Without this, SAC spends a long time just learning to follow a road at all.
3. The planner's transitions (with rewards) pre-fill the replay buffer for the critics.
4. SAC fine-tunes with the env reward (progress, lane error, smoothness, shield interventions,
   collisions). The safety shield stays inside the env, exactly as on the car.
Checkpoints every 50k steps; evaluation on fixed seeds is done by rl/evaluate.py.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from kin_env import KinematicDrivingEnv, planner_action  # noqa: E402
from stable_baselines3 import SAC  # noqa: E402
from stable_baselines3.common.callbacks import CheckpointCallback  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402


def collect_demos(env, steps, seed=0):
    obs_l, act_l, rew_l, next_l, done_l = [], [], [], [], []
    obs, _ = env.reset(seed=seed)
    prev, ep = 0.0, 0
    t0 = time.time()
    for i in range(steps):
        a = planner_action(env, prev)
        prev = a[0] * 0.45
        nobs, r, term, trunc, _ = env.step(a)
        obs_l.append(obs), act_l.append(a), rew_l.append(r), next_l.append(nobs), done_l.append(term)
        obs = nobs
        if term or trunc:
            ep += 1
            obs, _ = env.reset(seed=seed + ep)
            prev = 0.0
        if (i + 1) % 5000 == 0:
            print(f'  demos {i + 1}/{steps} ({ep} episodes, {time.time() - t0:.0f}s)', flush=True)
    return (np.array(obs_l), np.clip(np.array(act_l), -0.999, 0.999), np.array(rew_l, np.float32),
            np.array(next_l), np.array(done_l))


def behaviour_clone(model, obs, act, epochs=30, batch=256, lr=1e-3):
    actor = model.policy.actor
    opt = torch.optim.Adam(actor.parameters(), lr=lr)
    dev = model.device
    O = torch.as_tensor(obs, dtype=torch.float32, device=dev)
    A = torch.as_tensor(act, dtype=torch.float32, device=dev)
    n = len(O)
    for ep in range(epochs):
        perm = torch.randperm(n, device=dev)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            mean_actions, log_std, _ = actor.get_action_dist_params(O[idx])
            pred = torch.tanh(mean_actions)
            loss = ((pred - A[idx]) ** 2).mean() + 1e-3 * (log_std ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if ep % 5 == 0 or ep == epochs - 1:
            print(f'  BC epoch {ep:2d}: action MSE {total / n:.4f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=300000)
    ap.add_argument('--demo-steps', type=int, default=30000)
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'sac_driver'))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    env = Monitor(KinematicDrivingEnv(seed=0))
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=400000, batch_size=256, gamma=0.99,
                tau=0.005, ent_coef='auto', learning_starts=0, train_freq=1, gradient_steps=1,
                policy_kwargs=dict(net_arch=[256, 256]), verbose=0, seed=0)
    print(f'device: {model.device}', flush=True)

    print('1) planner demonstrations', flush=True)
    demo = collect_demos(env.unwrapped, a.demo_steps)
    print('2) behaviour cloning', flush=True)
    behaviour_clone(model, demo[0], demo[1])
    model.save(a.out + '_bc')
    print('3) replay buffer pre-fill with planner transitions', flush=True)
    for o, act, r, no, d in zip(*demo):
        model.replay_buffer.add(o[None], no[None], act[None], np.array([r]), np.array([d]), [{}])
    print('4) SAC fine-tuning', flush=True)
    ckpt = CheckpointCallback(save_freq=50000, save_path=os.path.join(HERE, 'models', 'ckpt'), name_prefix='sac')
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=ckpt, log_interval=20, progress_bar=False)
    model.save(a.out)
    print(f'done in {(time.time() - t0) / 60:.0f} min -> {a.out}.zip', flush=True)


if __name__ == '__main__':
    main()
