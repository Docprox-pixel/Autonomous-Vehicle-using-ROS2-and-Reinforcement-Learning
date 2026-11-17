#!/usr/bin/env python3
"""Train the give-way DECISION policy (YIELD / CREEP / GO) with PPO in rl/decision_env.py.

  python3 rl/train_decision.py [--steps 800000]  -> rl/models/giveway_ppo.zip
  python3 rl/eval_decision.py --policy rl/models/giveway_ppo.zip
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from decision_env import GiveWayEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=800000)
    ap.add_argument('--out', default=os.path.join(HERE, 'models', 'giveway_ppo.zip'))
    a = ap.parse_args()
    import torch
    torch.set_num_threads(2)                 # 8 env processes + a small net: threads only contend
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import SubprocVecEnv
    env = make_vec_env(lambda: GiveWayEnv(), n_envs=8, seed=7, vec_env_cls=SubprocVecEnv)
    model = PPO('MlpPolicy', env, n_steps=512, batch_size=1024, n_epochs=8, gamma=0.995, gae_lambda=0.95,
                ent_coef=0.01, learning_rate=3e-4, policy_kwargs={'net_arch': [128, 128]}, device='cpu', verbose=0)
    chunk = 100000
    for k in range(0, a.steps, chunk):
        model.learn(chunk, reset_num_timesteps=False)
        model.save(a.out)
        print(f'{k + chunk} steps saved -> {a.out}', flush=True)


if __name__ == '__main__':
    main()
