import gym
from stable_baselines3 import PPO
import os
from custom_envs.games import snake_env

env = snake_env() # continuous: LunarLanderContinuous-v2
env.reset()

model_path = os.path.join("models","1644505910","80000.zip")

model = PPO.load(model_path, env=env)

episodes = 50

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        print(rewards)