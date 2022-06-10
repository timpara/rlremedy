from rlremedy.envs.time_series.plain_vanilla import time_series_env
from rlremedy.envs.games.snake import snake_env
from rlremedy.envs.time_series.multi_asset import multi_asset_env
import pytest


slow_test = pytest.mark.skipif(
    "not config.getoption('--run-slow')",
    reason="Only run when --run-slow is given",
)
@slow_test
def test_time_series_env_random_action():
    '''
    Instantiates the time series env and samples random action.
    Allows for a quick end-to-end check
    '''
    env = time_series_env()
    episodes = 5

    for _ in range(episodes):
        env.reset()
        while not env.done:
            print("------------------")
            random_action = env.action_space.sample()
            print("action",random_action)
            obs, total_reward, _, info = env.step(random_action)
            print('reward',total_reward)


@slow_test
def test_snake_env_random_action():
    '''
    Instantiates the snake env and samples random action.
    Allows for a quick end-to-end check
    '''
    env = snake_env()
    episodes = 5
    env.reset()
    for _ in range(episodes):
        while not env.done:
            random_action = env.action_space.sample()
            print("action",random_action)
            obs, reward, done, info = env.step(random_action)
            print('reward',reward)

@slow_test
def test_multi_asset_random_action():
    '''
    Instantiates the snake env and samples random action.
    Allows for a quick end-to-end check
    '''
    env = multi_asset_env()
    episodes = 5
    env.reset()
    for _ in range(episodes):
        while not env.done:
            random_action = env.action_space.sample()
            print("action",random_action)
            obs, reward, done, info = env.step(random_action)
            print('reward',reward)
