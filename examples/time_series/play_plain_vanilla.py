from stable_baselines3 import DQN
from rlremedy.envs.time_series import time_series_env
import os


env = time_series_env() # continuous: LunarLanderContinuous-v2
env.reset()

model_path = os.path.join("logs","DQN","best_model.zip")

model = DQN.load(model_path, env=env)

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