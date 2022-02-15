from stable_baselines3.common.env_checker import check_env
from custom_envs.time_series import time_series_env
from custom_envs.games import snake_env

import pytest

#%%
@pytest.mark.integration
def test_time_series_env():
    '''
    Instantiates the env and feeds it into the env_check. Output is generated if sanity checks fail.
    :return:
    '''
    env = time_series_env()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

@pytest.mark.integration
def test_time_series_env():
    '''
    Instantiates the snake env and feeds it into the env_check. Output is generated if sanity checks fail.
    :return:
    '''
    env = snake_env()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)