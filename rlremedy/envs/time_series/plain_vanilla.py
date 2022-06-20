import gym
from gym import spaces
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from gym.envs.registration import register
from rlremedy.models import time_series
from rlremedy.models.time_series import OuProcess
from rlremedy.models.time_series import configuration as process_conf

from sklearn.preprocessing import minmax_scale
plt.ion()
class DynamicUpdate():
    #Suppose we know the x range
    min_x = 0
    max_x = 1

    def on_launch(self):
        #Set up plot
        self.figure, self.ax = plt.subplots()
        self.lines, = self.ax.plot([],[], 'o')
        #Autoscale on unknown axis and known lims on the other
        self.ax.set_autoscaley_on(True)
        self.ax.set_xlim(self.min_x, self.max_x)
        #Other stuff
        self.ax.grid()


    def on_running(self, xdata, ydata):
        #Update data (with the new _and_ the old points)
        self.lines.set_xdata(xdata)
        self.lines.set_ydata(ydata)
        #Need both of these in order to rescale
        self.ax.relim()
        self.ax.autoscale_view()
        #We need to draw *and* flush
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()




class time_series_env(gym.Env):

    def __init__(self,
                 data_generating_process = OuProcess,
                 process_params = process_conf.OUParams):
        super(time_series_env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)#buy sell neutral
        # Number of past ticks per feature to be used as observations (1440min=1day, 10080=1Week, 43200=1month, )
        self.obs_ticks = 1
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_ticks*4, 1),
                                            dtype=float)

        self.process_params = process_params
        self.reward_range = (-float("inf"), float("inf"))
        self.data_generating_process = data_generating_process(self.process_params)
    def register(self,env_id):
        register(
            # unique identifier for the env `name-version`
            id=env_id,
            # path to the class for creating the env
            # Note: entry_point also accept a class as input (and not only a string)
            entry_point=time_series_env,  # time_series_env,
            # Max number of steps per episode, using a `TimeLimitWrapper`
            max_episode_steps=500,
        )

    def step(self, action):
        self.tick_count += 1
        # On last timestep kill the step
        if self.tick_count == len(self.data)-1:
            self.done = True



        self.all_previous_actions.append(action)

        # create observation:
        self.observation = self._next_observation()





        # max sum sofar
        if self.tick_count >= 2:
            reward = self._take_action(prev_action=self.prev_actions)
            max_return = np.sum(np.abs(np.diff(self.data[:self.tick_count + 1].ravel())))
            std_max_return = np.std(np.abs(np.diff(self.data[:self.tick_count + 1].ravel())))
            max_sharpe = np.sum(max_return)/std_max_return
            self.total_reward = reward / max_sharpe
        else:
            self.total_reward = 0

        info = {"Total_reward": self.total_reward, "tick_count": self.tick_count}

        self.prev_actions.append(action)

        return self.observation, self.total_reward, self.done, info

    def _next_observation(self):
        diff_market=np.diff(self.data[max(self.tick_count - 5,0):self.tick_count+1],axis=0)
        market_state = np.zeros(1) if np.all(diff_market!=np.NaN) else np.nanmean(diff_market)
        market_state = np.append(market_state,self.data[self.tick_count])
        observation = np.append(market_state, np.append(self.prev_actions[-1], self.total_reward))
        return observation.reshape(self.obs_ticks*4,-1).astype(np.float64)


    def reset(self):
        # Initial action

        self.prev_actions = deque(maxlen=1)
        self.prev_reward = 0
        self.data_at_step = deque(maxlen=2)
        self.tick_count = 2
        self.done = False
        self.all_previous_actions = []
        #create toy sin data
        # 10k linearly spaced numbers
        self.total_reward = 0
        self.rewards = []
        self.ticks = self.process_params.sample_size
        self.data = self.data_generating_process.sample_paths()

        self._first_rendering=True
        for _ in range(self.prev_actions.maxlen):
            self.prev_actions.append(0)  # to create history
        for _ in range(self.data_at_step.maxlen):
            self.data_at_step.append(0)

        # create observation:
        observation = self._next_observation()

        return observation

    def render(self, mode="human"):
        if self.tick_count < 2:
            return


        def _plot_position():

            if self.all_previous_actions[-1]==0:
                color = 'green'
            elif self.all_previous_actions[-1]==1:
                color = 'red'
            elif self.all_previous_actions[-1] == 2:
                color = 'yellow'
            #elif self.all_previous_actions[-1]==0:

            #if color:
            plt.scatter(self.tick_count,self.data[self.tick_count-1], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.figure(figsize=(18.5,10.5))
            plt.cla()
            plt.plot(self.data)
            _plot_position()
        #plt.plot(self.data)
        _plot_position()

        plt.suptitle(
            f"Total Reward: {np.round(self.total_reward,2)}")


    def _take_action(self,prev_action):

        if prev_action[-1] == 0:
            # we buy
            reward = (self.data[self.tick_count] - self.data[self.tick_count-1])

        elif prev_action[-1] == 1:
            # we sell
            reward = (self.data[self.tick_count-1]- self.data[self.tick_count])

        elif prev_action[-1] == 2:
            reward = 0
            # all flat
        reward = float(reward)


        self.rewards.append(reward)
        if len(self.rewards)>1:
            sharpe = np.sum(self.rewards)/np.std(self.rewards)
            return sharpe
        else:
            return 0
    def pause_rendering(self):
        sharpe = np.sum(self.rewards) / np.nanstd(self.rewards)
        print(f"Sharpe Ratio: {sharpe}")
        plt.show()
        return sharpe