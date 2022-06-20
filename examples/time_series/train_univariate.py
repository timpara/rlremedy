from stable_baselines3 import PPO
import os
from rlremedy.envs.time_series.plain_vanilla import time_series_env

from stable_baselines3.common.evaluation import evaluate_policy
from rlremedy.models.time_series import configuration as process_conf
from rlremedy.models.time_series import SeasonalProcess as data_generating_process
from stable_baselines3.common.monitor import Monitor
from rlremedy.learning.utils import linear_schedule
import gym
import tensorflow as tf
from utils.callbacks import SaveOnBestTrainingRewardCallback
import wandb
from rlremedy.policies.actor_critic import CustomActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnMaxEpisodes

models_dir = os.path.join("models","univariate")#, str(int(time.time())))
logdir = os.path.join("logs","univariate")# str(int(time.time())))
env_id = "TimeSeries-OU-v1"
max_timesteps = 5000
max_iterations = 100

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

process_params = process_conf.SPParams()#OUParams(alpha=0.6, gamma=0, beta=1, sample_size=max_timesteps)
vec_env = time_series_env(data_generating_process=data_generating_process,
                          process_params=process_params)

vec_env.register(env_id=env_id)
vec_env = gym.make(env_id)
# Because we use parameter noise, we should use a MlpPolicy with layer normalization

vec_env = Monitor(vec_env, logdir)

# num_cpu = 10  # Number of processes to use
# Optional Create the vectorized environment
# vec_env = make_vec_env(env_id, n_envs=num_cpu)
# vec_env = time_series_env()
wandb.init(
    config={
        "policy": 'MlpLstmPolicy',
        "total_max_timesteps": max_timesteps * max_iterations},
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="PPO-Toy-Example",
    monitor_gym=True,  # automatically upload gym environements' videos
    save_code=True)

model = PPO(CustomActorCriticPolicy, vec_env, verbose=1,tensorboard_log=logdir,learning_rate=linear_schedule(0.00001))

#model = PPO("MlpLstmPolicy",
#            vec_env,
#            verbose=1,
#            tensorboard_log=logdir,
#            seed=2)
            #policy_kwargs=dict(activation_fn=th.nn.ReLU,
            #         net_arch=[dict(pi=[128,1258], vf=[128,128])]))

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
callback = [SaveOnBestTrainingRewardCallback(check_freq=5000, log_dir=logdir),
         EvalCallback(vec_env, best_model_save_path='./logs/',
                             log_path=logdir, eval_freq=5000,
                             deterministic=True, render=False),
            StopTrainingOnMaxEpisodes(max_episodes=500, verbose=1)]
iters = 0
#while iters < max_iterations:
#    iters += 1
model.learn(total_timesteps=max_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"PPO_avg_sharpe",
            callback=callback)
    #model.save(os.path.join(models_dir,str(max_timesteps*iters)))
wandb.finish()
