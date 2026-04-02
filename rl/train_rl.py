import os
import sys

# Must be run from the rl/ directory
sys.path.insert(0, os.path.dirname(__file__))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from car_env import CarEnv


# -------- SINGLE ENV (No eval env conflict) --------
def make_env():
    return Monitor(CarEnv())

env = DummyVecEnv([make_env])


# -------- CHECKPOINT --------
checkpoint_callback = CheckpointCallback(
    save_freq=5000,
    save_path="./ppo_checkpoints/",
    name_prefix="bmw_model"
)


# -------- LOAD / CREATE MODEL --------
model_path = "bmw_rl_driver.zip"

if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    model = PPO.load(model_path, env=env)
else:
    print("Creating new PPO model...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,       # Higher LR = faster learning
        n_steps=512,              # Smaller rollout = updates more often
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.05,            # High entropy = more exploration forward
        tensorboard_log="./ppo_logs/",
    )


# -------- TRAIN --------
print("Training started... Car should move forward!")
print("Press Ctrl+C to stop and save")

try:
    model.learn(
        total_timesteps=500000,
        callback=[checkpoint_callback]
    )
except KeyboardInterrupt:
    print("Stopped by user.")


# -------- SAVE --------
model.save("bmw_rl_driver")
print("Model saved as bmw_rl_driver.zip")
print("Training complete!")