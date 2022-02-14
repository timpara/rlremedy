from stable_baselines3.common.env_checker import check_env
from custom_envs.time_series import time_series_env
import pytest

#%%
@pytest.mark.integration
def test_time_series_env():

    env = time_series_env()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)