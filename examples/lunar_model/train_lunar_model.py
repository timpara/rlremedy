import gym
from stable_baselines3 import PPO
import os
models_dir = "models/PPO"
logdir = "logs"


env = gym.make('LunarLander-v2')
env.reset()

model = PPO('MlpPolicy', env, verbose=1)

if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

iters = 0
TIMESTEPS = 10000
while True:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")


