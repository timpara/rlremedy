from stable_baselines3 import PPO
import os
from rlremedy.envs.time_series.multi_asset import multi_asset_env
import time
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from utils.callbacks import SaveOnBestTrainingRewardCallback
import wandb


env_id = "MultiAsset-v1"
models_dir = os.path.join("models",env_id,str(int(time.time())))
logdir = os.path.join("logs",env_id,str(int(time.time())))
TIMESTEPS = 252
max_iters=1e4


if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

vec_env = multi_asset_env()
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
    project="DDPG-MultiAsset-Example",
    monitor_gym=True,       # automatically upload gym environements' videos
    save_code=False)

model = PPO("MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log=logdir,seed=2,batch_size=10)


# Random Agent, before training
mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=10)
callback = SaveOnBestTrainingRewardCallback(check_freq=TIMESTEPS, log_dir=logdir)


iters = 0
while iters<max_iters:
    iters += 1
    model.learn(total_timesteps=TIMESTEPS,
            reset_num_timesteps=False,
            tb_log_name=f"DDPG",
            callback=callback,
            log_interval=100)
    #model.save(os.path.join(models_dir,str(TIMESTEPS*iters)))
wandb.finish()