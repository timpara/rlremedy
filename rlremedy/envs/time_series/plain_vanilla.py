import gym
from gym import spaces
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from gym.envs.registration import register
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

    def __init__(self):
        super(time_series_env, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(2)#buy sell
        # Number of past ticks per feature to be used as observations (1440min=1day, 10080=1Week, 43200=1month, )
        self.obs_ticks = 3
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.obs_ticks, 1),
                                            dtype=int)
        self.total_reward = 0
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
        if self.tick_count == len(self.my_data)-1:
            self.done = True



        self.all_previous_actions.append(action)

        # create observation:
        self.observation = self._next_observation()




        self._take_action(prev_action=self.prev_actions.pop(),current_action=action,data=self.data_at_step)
        self.total_reward += self.reward

        info = {"Total_reward": self.total_reward, "tick_count": self.tick_count}

        self.prev_actions.append(action)

        return self.observation, self.total_reward, self.done, info

    def _next_observation(self):
        self.data_at_step.append(self.my_data[self.tick_count, 0])
        observation = np.array([int(np.where(np.diff(self.data_at_step)<0,-1,1)), self.prev_actions[-1], self.total_reward]).ravel()
        observation = np.reshape(observation, (self.obs_ticks,1 ))
        return observation

    def reset(self):
        # Initial action

        self.prev_actions = deque(maxlen=1)
        self.prev_reward = 0
        self.data_at_step = deque(maxlen=2)
        self.tick_count = 0
        self.done = False
        self.all_previous_actions = []
        #create toy sin data
        # 10k linearly spaced numbers

        self.ticks = np.arange(0, 1000)
        cycles = 5  # how many sine cycles
        resolution = 1000  # how many datapoints to generate

        length = np.pi * 2 * cycles
        self.my_data = np.reshape(np.sin(np.arange(0, length, length / resolution)),[-1,1])
        self.my_data = minmax_scale(self.my_data, feature_range=(0, 1))



        self._first_rendering=True
        for _ in range(self.prev_actions.maxlen):
            self.prev_actions.append(0)  # to create history
        for _ in range(self.data_at_step.maxlen):
            self.data_at_step.append(0)

        # create observation:
        observation = self._next_observation()

        return observation

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


    def _take_action(self,prev_action,current_action,data):

        if prev_action ==0:
            # we buy
            self.reward = (data[1] - data[0])

        elif prev_action == 1:
            # we sell
            self.reward = (data[0] - data[1])
        else:
            self.reward = 0
            # all flat
        if current_action != prev_action:
            self.reward=-abs(self.reward)

    def pause_rendering(self):
        plt.show()