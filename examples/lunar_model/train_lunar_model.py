import gym
from stable_baselines3 import PPO
import os
from utils.callbacks import SaveOnBestTrainingRewardCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

models_dir = "models/lunar"
logdir = "logs/lunar"

def make_env():
    env = gym.make('LunarLander-v2')
    env = Monitor(env)  # record stats such as returns
    return env

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/lunar", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)


model = PPO('MlpPolicy', env, verbose=1)
iters = 0
max_iters=100
TIMESTEPS = 1000
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

wandb.init(
    config={
    "policy": 'MlpPolicy'},
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="PPO-Lunar",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=False)
callback = [SaveOnBestTrainingRewardCallback(check_freq=TIMESTEPS, log_dir=logdir),WandbCallback(),
            ]
while iters<max_iters:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False,callback=callback)
    model.save(f"{models_dir}/{TIMESTEPS*iters}")
wandb.finish()

