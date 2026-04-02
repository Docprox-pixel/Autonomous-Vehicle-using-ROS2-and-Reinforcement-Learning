import os
import time
from stable_baselines3 import PPO
from car_env import CarEnv

def main():
    # 1. Create the environment
    env = CarEnv()
    
    # 2. Load the trained model
    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, "bmw_rl_driver")
    
    try:
        model = PPO.load(model_path)
        print("Model loaded successfully from:", model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have trained the model first and bmw_rl_driver.zip exists.")
        return

    # 3. Run the model in a loop
    obs, info = env.reset()
    while True:
        # Predict action based on current observation
        action, _states = model.predict(obs, deterministic=True)
        
        # Take step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Display live telemetry directly to the RL terminal
        out = (f"\r🤖 RL AGENT | Speed: {env.speed*15.0:4.1f} m/s | Steer: {float(action[1]):5.2f} "
               f"| Lane Err: {env.lane_error:5.2f} | Objects: {env.objects_detected:<10} | Reward: {reward:5.1f}")
        print(out, end="", flush=True)
        
        if terminated or truncated:
            print("\nEpisode finished. Resetting environment...")
            obs, info = env.reset()
            time.sleep(1.0)

if __name__ == "__main__":
    main()
