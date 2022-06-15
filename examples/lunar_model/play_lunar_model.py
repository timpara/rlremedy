import gym
import os
from stable_baselines3 import PPO

models_dir = f"models/lunar/"

env = gym.make('LunarLander-v2')  # continuous: LunarLanderContinuous-v2
env.reset()

model_path = os.path.join(models_dir,"best_model")
model = PPO.load(model_path, env=env)

episodes = 5

for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
        print(rewards)