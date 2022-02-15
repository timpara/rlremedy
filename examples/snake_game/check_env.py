from stable_baselines3.common.env_checker import check_env
from custom_envs.games import snake_env

env = snake_env()
# It will check your custom environment and output additional warnings if needed
check_env(env)