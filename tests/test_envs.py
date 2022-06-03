from stable_baselines3.common.env_checker import check_env
from rlremedy.envs.time_series.plain_vanilla import time_series_env
from rlremedy.envs.games.snake import snake_env
from rlremedy.envs.time_series.multi_asset import multi_asset_env
import pytest

#%%
@pytest.mark.integration
def test_plain_vanilla_env():
    '''
    Instantiates the env and feeds it into the env_check. Output is generated if sanity checks fail.
    :return:
    '''
    env = time_series_env()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

@pytest.mark.integration
def test_snake_env():
    '''
    Instantiates the snake env and feeds it into the env_check. Output is generated if sanity checks fail.
    :return:
    '''
    env = snake_env()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

@pytest.mark.integration
def test_multi_asset():
    '''
    Instantiates the snake env and feeds it into the env_check. Output is generated if sanity checks fail.
    :return:
    '''
    env = multi_asset_env()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)