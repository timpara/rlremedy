from stable_baselines3 import PPO
from rlremedy.envs.time_series.multi_asset import multi_asset_env
import os


env = multi_asset_env() # continuous: LunarLanderContinuous-v2
env.reset()

model_path = os.path.join("logs","MultiAsset-v1","1655121381","best_model.zip")

model = PPO.load(model_path, env=env)

episodes = 1

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)
    env.pause_rendering()