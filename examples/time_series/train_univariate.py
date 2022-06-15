from stable_baselines3 import PPO
import os
from rlremedy.envs.time_series.plain_vanilla import time_series_env
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from rlremedy.models.time_series import configuration as process_conf
from rlremedy.models.time_series import OuProcess
from rlremedy.models.time_series import SeasonalProcess
from utils.callbacks import SaveOnBestTrainingRewardCallback
import wandb

models_dir = os.path.join("models", str(int(time.time())))
logdir = os.path.join("logs", str(int(time.time())))
env_id = "TimeSeries-v1"
max_timesteps = 1000
max_iterations = 1e4

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

vec_env = time_series_env(data_generating_process=OuProcess,
                          process_params=process_conf.OUParams)

vec_env.register(env_id=env_id)
vec_env = Monitor(vec_env, logdir)

# num_cpu = 10  # Number of processes to use
# Optional Create the vectorized environment
# vec_env = make_vec_env(env_id, n_envs=num_cpu)
# vec_env = time_series_env()
wandb.init(
    config={
        "policy": 'MlpPolicy',
        "total_max_timesteps": max_timesteps * max_iterations},
    sync_tensorboard=True,  # automatically upload SB3's tensorboard metrics to W&B
    project="DQN-Toy-Example",
    monitor_gym=True,  # automatically upload gym environements' videos
    save_code=True)

model = PPO("MlpPolicy",
            vec_env,
            verbose=1,
            batch_size=1000,
            tensorboard_log=logdir,
            seed=2)

# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=logdir)

iters = 0
while iters < max_iterations:
    iters += 1
    model.learn(total_timesteps=max_timesteps,
                reset_num_timesteps=False,
                tb_log_name=f"PPO",
                callback=callback)
    model.save(os.path.join(models_dir,str(max_timesteps*iters)))
wandb.finish()
