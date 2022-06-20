from stable_baselines3 import PPO
from rlremedy.envs.time_series.plain_vanilla import time_series_env
from rlremedy.models.time_series import configuration as process_conf
from rlremedy.models.time_series import SeasonalProcess as data_generating_process
import os

process_params = process_conf.SPParams()#.OUParams(alpha=0.6, gamma=0, beta=0.1, sample_size=500)

env = time_series_env(data_generating_process=data_generating_process,
                          process_params=process_params)



model_path = os.path.join("logs","univariate","best_model.zip")

model = PPO.load(model_path, env=env)

episodes = 2
sharpes=[]
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    env.pause_rendering()
