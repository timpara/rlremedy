from stable_baselines3.common.env_checker import check_env
from rlremedy.envs.time_series import plain_vanilla
from rlremedy.envs.games import snake_env

import pytest

#%%
@pytest.mark.integration
def test_plain_vanilla_env():
    '''
    Instantiates the env and feeds it into the env_check. Output is generated if sanity checks fail.
    :return:
    '''
    env = plain_vanilla()
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