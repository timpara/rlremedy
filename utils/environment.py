from typing import Callable
from stable_baselines3.common.utils import set_random_seed
import gym


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    To multiprocess RL training, we will just have to wrap the Gym env into a SubprocVecEnv object,
    that will take care of synchronising the processes. The idea is that each process will run an indepedent instance of the Gym env.
    For that, we need an additional utility function, make_env, that will instantiate the environments and make
    sure they are different (using different random seed).
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init