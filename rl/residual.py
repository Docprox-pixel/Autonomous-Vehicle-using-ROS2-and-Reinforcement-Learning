#!/usr/bin/env python3
"""Residual RL: the policy learns CORRECTIONS on top of the hand-written planner.

    action_final = clip(planner_action + RESIDUAL_SCALE * policy_action)

Why: the first SAC policy (driving from scratch) reached 10% success in the training env and froze
in Webots, where the scene (buildings, poles, kerbs) differs from training. A residual policy starts
exactly as good as the planner (zero correction) and can only learn to improve on it, e.g. smoother
speed in turns, earlier braking behind lead cars, better lane centring. It transfers to Webots
because the planner, which already works there, does the heavy lifting.

Observation = the normal 50-dim observation (kin_env.obs_from) + the planner's own action (2).

  python3 rl/residual.py --steps 150000             -> rl/models/residual_driver.zip
  python3 rl/evaluate.py --policy rl/models/residual_driver.zip --residual
"""
import argparse
import os
import sys
import time

import gymnasium as gym
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, HERE)
from kin_env import KinematicDrivingEnv, planner_action  # noqa: E402

RESIDUAL_SCALE = 0.3          # max correction: 0.3 * 0.45 rad steering, 0.3 * 4 m/s speed


def combine(base, residual):
    return np.clip(np.asarray(base, np.float32) + RESIDUAL_SCALE * np.asarray(residual, np.float32), -1.0, 1.0)


class ResidualEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        lo = np.r_[env.observation_space.low, [-1, -1]].astype(np.float32)
        hi = np.r_[env.observation_space.high, [1, 1]].astype(np.float32)
        self.observation_space = gym.spaces.Box(lo, hi, dtype=np.float32)
        self.prev_steer = 0.0

    def _aug(self, obs):
        self.base = planner_action(self.env.unwrapped, self.prev_steer)
        return np.r_[obs, self.base].astype(np.float32)

    def reset(self, **kw):
        obs, info = self.env.reset(**kw)
        self.prev_steer = 0.0
        return self._aug(obs), info

    def step(self, action):
        a = combine(self.base, action)
        self.prev_steer = float(a[0]) * 0.45
        obs, r, term, trunc, info = self.env.step(a)
        r -= 0.2 * float(np.abs(action).sum())          # small cost for deviating from the planner
        info['final_action'] = a
        return self._aug(obs), r, term, trunc, info


def main():
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=150000)
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'residual_driver'))
    a = ap.parse_args()
    env = Monitor(ResidualEnv(KinematicDrivingEnv(seed=0)))
    # Start with a tiny, nearly-deterministic correction (log_std -3): the car drives like the
    # planner from the first step, so exploration does not wreck the replay buffer with crashes.
    model = SAC('MlpPolicy', env, learning_rate=3e-4, buffer_size=300000, batch_size=256, gamma=0.99,
                tau=0.005, ent_coef='auto_0.05', learning_starts=2000, train_freq=1, gradient_steps=1,
                policy_kwargs=dict(net_arch=[256, 256], log_std_init=-3.0), verbose=0, seed=0)
    print(f'device: {model.device}', flush=True)
    ckpt = CheckpointCallback(save_freq=25000, save_path=os.path.join(HERE, 'models', 'ckpt_residual'),
                              name_prefix='residual')
    t0 = time.time()
    model.learn(total_timesteps=a.steps, callback=ckpt, log_interval=20, progress_bar=False)
    model.save(a.out)
    print(f'done in {(time.time() - t0) / 60:.0f} min -> {a.out}.zip', flush=True)


if __name__ == '__main__':
    main()
