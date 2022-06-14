import gym
from stable_baselines3 import PPO
import os
from rlremedy.envs.time_series.multi_asset import multi_asset_env
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from utils.callbacks import SaveOnBestTrainingRewardCallback
import wandb
from wandb.integration.sb3 import WandbCallback

env_id = "MultiAsset-v1"
models_dir = os.path.join("models",env_id,str(int(time.time())))
logdir = os.path.join("logs",env_id,str(int(time.time())))
TIMESTEPS = 252
max_iters=1e3
batch_size = 2048
n_eval_episodes=10
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

vec_env = multi_asset_env()
vec_env.register(env_id=env_id,max_episode_steps=TIMESTEPS)

def make_env():
    env = gym.make(env_id)
    env = Monitor(env,logdir)  # record stats such as returns
    return env

vec_env = gym.make(env_id)

vec_env = DummyVecEnv([make_env])
#vec_env = VecVideoRecorder(vec_env, f"videos/{env_id}", record_video_trigger=lambda x: x % 1000 == 0, video_length=200)
#vec_env = make_vec_env(lambda: multi_asset_env(), n_envs=1)

wandb.init(
    config={
    "policy": 'MlpPolicy',
    "total_timesteps": TIMESTEPS*max_iters},
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="PPO-MultiAsset-ActionCosts",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=False)

model = PPO("MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=logdir,seed=2,batch_size=batch_size)


# Random Agent, before training
#mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=n_eval_episodes)
callback = [SaveOnBestTrainingRewardCallback(check_freq=TIMESTEPS, log_dir=logdir),WandbCallback(
        gradient_save_freq=100,
        model_save_path=logdir,
        verbose=2,
    ),
            ]
'''
wandb.config.learning_rate=model.learning_rate
wandb.config.batch_size=batch_size
wandb.config.n_eval_episodes=n_eval_episodes
wandb.config.num_timesteps=model.num_timesteps
'''




iters = 0
#wandb.watch
while iters<max_iters:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            callback=callback,
            log_interval=1000)
    if iters % n_eval_episodes == 0:
        mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=n_eval_episodes)
        wandb.log({"mean_reward":mean_reward})
        wandb.log({"std_reward": std_reward})
    model.save(os.path.join(logdir))
wandb.finish()