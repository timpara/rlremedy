import gym
from gym import spaces
import numpy as np
import cv2
import random
import time
from collections import deque



class time_series_env(gym.Env):

    def __init__(self):
        super(time_series_env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)#buy sell neutral
        # Number of past ticks per feature to be used as observations (1440min=1day, 10080=1Week, 43200=1month, )
        self.obs_ticks = 10
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=float(-1.0), high=float(1.0), shape=(self.obs_ticks, 1), dtype=np.float32)
        self.total_reward = 0

    def step(self, action):
        self.prev_actions.append(action)

        self.data_at_step.append(self.my_data[self.tick_count,1])



        # Change the head position based on the button direction
        if action == 1:
            # we buy
            print("Buy")
            self.open_trade = 1
            self.reward =  self.data_at_step[0]-self.data_at_step[1]
        elif action == 2:
            # we sell
            print("Sell")
            self.reward = self.data_at_step[1] - self.data_at_step[0]
        elif action == 0:
            print("Flat")
            self.reward = 0
            # all flat


        # On last timestep kill the step
        if self.tick_count == len(self.my_data):
            self.done = True



        self.total_reward += self.reward



        # create observation:
        info = {"Total_reward": self.total_reward, "tick_count": self.tick_count}
        observation = [self.data_at_step[0], self.data_at_step[1]] + list(self.prev_actions)
        observation = np.array(observation)
        self.tick_count +=1
        return observation, self.total_reward, self.done, info

    def reset(self):
        # Initial action

        self.prev_actions = deque(maxlen=2)
        self.prev_reward = 0
        self.data_at_step = deque(2)
        self.tick_count = 0
        self.done = False

        #create toy sin data
        self.my_data = np.reshape(np.sin(np.random.rand(1000) * 2),[-1,1])

        # however long we aspire the snake to be
        for _ in range(2):
            self.prev_actions.append(-1)  # to create history
            self.data_at_step.append(0)
        # create observation:
        observation = [self.data_at_step[0], self.data_at_step[1]] + list(self.prev_actions)
        observation = np.array(observation)

        return observation