import gym
from gym import spaces
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from gym.envs.registration import register
from sklearn.preprocessing import minmax_scale
import numpy as np
import tensorflow as tf
from rlremedy.models.time_series.multivariate_geo_brownian_motion import MultivariateGeometricBrownianMotion
from utils.math.random_ops import RandomType
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




class multi_asset_env(gym.Env):

    def __init__(self):
        super(multi_asset_env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        # Number of past ticks per feature to be used as observations (1440min=1day, 10080=1Week, 43200=1month, )
        self.obs_ticks = 1
        self.means = [-0.01, 0.02]
        self.number_of_assets = len(self.means)
        self.volatilities = [0.1, 0.2]
        self.corr_matrix = [[1, 0.1], [0.1, 1]]
        self.times =  tf.linspace(tf.constant(0.0, dtype=np.float64), 252, 253)
        self.num_samples_local=1
        self.initial_state = [1.0, 2.0]
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=float(-1), high=float(1e4), shape=(self.obs_ticks,self.number_of_assets*2+1), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0,high=1.0,shape=(1,self.number_of_assets),dtype=np.float16)#buy sell
        self.seed_sampling=[1,2]
        self.total_reward = 0
    def register(self,env_id):
        register(
            # unique identifier for the env `name-version`
            id=env_id,
            # path to the class for creating the env
            # Note: entry_point also accept a class as input (and not only a string)
            entry_point=multi_asset_env,  # time_series_env,
            # Max number of steps per episode, using a `TimeLimitWrapper`
            max_episode_steps=500,
        )

    def step(self, action):
        self.tick_count += 1
        # On last timestep kill the step
        if self.tick_count == len(self.times)-1:
            self.done = True



        self.all_previous_actions.append(action)

        # create observation:
        self.observation = self._next_observation()
        self.total_reward += self._take_action(prev_action=self.prev_actions.pop())

        info = {"Total_reward": self.total_reward, "tick_count": self.tick_count}
        self.prev_actions.append(action)

        return self.observation, self.total_reward, self.done, info

    def _next_observation(self):
        observation = np.append(self.my_data[0, self.tick_count, :].numpy(), np.append(self.prev_actions[-1], self.total_reward))
        return observation.reshape(self.obs_ticks,-1).astype(np.float64)

    def reset(self):
        # Initial action

        self.prev_actions = deque(maxlen=1)
        self.prev_reward = 0.0
        self.tick_count = 0
        self.done = False
        self.all_previous_actions = []



        process = MultivariateGeometricBrownianMotion(
            dim=self.number_of_assets, means= self.means, volatilities= self.volatilities, corr_matrix= self.corr_matrix,
            dtype=tf.float64)



        self.my_data  = process.sample_paths(
            times=self.times, initial_state=self.initial_state,
            random_type=RandomType.STATELESS,
            num_samples=self.num_samples_local, normal_draws=None,seed=self.seed_sampling)



        self._first_rendering=True
        for _ in range(self.prev_actions.maxlen):
            self.prev_actions.append(np.zeros([1,self.number_of_assets]))  # to create history


        # create observation:
        # make observation to deque and append
        return self._next_observation()


    def render(self, mode="human"):
        if self.tick_count < 100:
            return
        def _plot_position():

            if self.all_previous_actions[-1]==0:
                color = 'green'
            elif self.all_previous_actions[-1]==1:
                color = 'red'
            else:
                color = 'yellow'
            #elif self.all_previous_actions[-1]==0:

            #if color:
            plt.scatter(self.tick_count,self.my_data[self.tick_count-1], color=color)

        if self._first_rendering:
            self._first_rendering = False
            plt.cla()
            plt.plot(self.my_data)
            _plot_position()
        plt.plot(self.my_data)
        _plot_position()

        plt.suptitle(
            "Total Reward: %.6f" % self.total_reward)


    def _take_action(self,prev_action):

        return_matrix = self.my_data[0, self.tick_count, :]-self.my_data[0, self.tick_count - self.obs_ticks, :]
        weighted_return_matrix=np.matmul(return_matrix, prev_action.T)
        return float(weighted_return_matrix)

    def pause_rendering(self):
        plt.show()