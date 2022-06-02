from stable_baselines3 import PPO
import os
from rlremedy.envs.games import snake_env
import time



models_dir = os.path.join("models",str(int(time.time())))
logdir = os.path.join("logs",str(int(time.time())))

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = snake_env()
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(os.path.join(models_dir,str(TIMESTEPS*iters)))