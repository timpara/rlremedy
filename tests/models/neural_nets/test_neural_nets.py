from stable_baselines3 import PPO
from rlremedy.policies.actor_critic import CustomActorCriticPolicy
import unittest

class TestDataProviderLocal(unittest.TestCase):

    def test_custom_actor_policy(self):
        model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1,seed=1)
        model.learn(5)
        assert model._last_obs.shape == (1,4)
        assert max(model._last_obs[0]) < 0.165
        assert min(model._last_obs[0]) > -0.33