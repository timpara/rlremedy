from custom_envs.time_series import time_series_env
import pytest

#%%
@pytest.mark.integration
def test_time_series_env():
    env = time_series_env()
    episodes = 50

    for _ in range(episodes):
        env.reset()
        while not env.done:
            print("------------------")
            random_action = env.action_space.sample()
            print("action",random_action)
            obs, total_reward, _, info = env.step(random_action)
            print('reward',total_reward)