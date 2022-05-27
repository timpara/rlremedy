from stable_baselines3 import PPO,DQN
import os
from custom_envs.time_series.plain_vanilla import time_series_env
import time
from stable_baselines3.common.env_util import make_vec_env
import gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from utils.callbacks import SaveOnBestTrainingRewardCallback
from utils.learning import linear_schedule
from utils.models import CustomActorCriticPolicy
import wandb

models_dir = os.path.join("models",str(int(time.time())))
logdir = os.path.join("logs",str(int(time.time())))
env_id = "TimeSeries-v1"
TIMESTEPS = 1000
max_iters=1e4


if not os.path.exists(models_dir):
	os.makedirs(models_dir)
if not os.path.exists(logdir):
	os.makedirs(logdir)

vec_env = time_series_env()
vec_env.register(env_id=env_id)
vec_env = Monitor(vec_env, logdir)

# num_cpu = 10  # Number of processes to use
# Optional Create the vectorized environment
# vec_env = make_vec_env(env_id, n_envs=num_cpu)
# vec_env = time_series_env()
wandb.init(
    config={
    "policy": 'MlpPolicy',
    "total_timesteps": TIMESTEPS*max_iters},
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="DQN-Toy-Example",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=True)
# model = PPO(CustomActorCriticPolicy, vec_env,verbose=1, tensorboard_log=logdir,learning_rate=linear_schedule(0.001))#policy_kwargs=policy_kwargs)

model = DQN("MlpPolicy",
            vec_env,
            verbose=1,
            train_freq=16,
            gradient_steps=8,
            gamma=0.99,
            exploration_fraction=0.2,
            exploration_final_eps=0.07,
            target_update_interval=600,
            learning_starts=100,
            buffer_size=10000,
            batch_size=128,
            learning_rate=4e-3,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=logdir,
            seed=2)


# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)


iters = 0
while iters<max_iters:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=f"PPO",
            callback=callback)
    #model.save(os.path.join(models_dir,str(TIMESTEPS*iters)))
wandb.finish()