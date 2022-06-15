from stable_baselines3 import PPO
import os
from rlremedy.envs.games.snake import snake_env
from utils.callbacks import SaveOnBestTrainingRewardCallback
from stable_baselines3.common.monitor import Monitor

import time



models_dir = os.path.join("models","snake",str(int(time.time())))
logdir = os.path.join("logs","snake",str(int(time.time())))

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = snake_env()
env = Monitor(env, logdir)
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 100
iters = 0
while True:
	iters += 1
	model.learn(total_timesteps=TIMESTEPS,
				reset_num_timesteps=False,
				tb_log_name=f"PPO",
				callback=SaveOnBestTrainingRewardCallback(check_freq=TIMESTEPS, log_dir=logdir))
	model.save(os.path.join(models_dir,str(TIMESTEPS*iters)))