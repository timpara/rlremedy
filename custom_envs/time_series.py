import gym
from gym import spaces
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

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
        self.action_space = spaces.Discrete(3)#buy sell neutral
        # Number of past ticks per feature to be used as observations (1440min=1day, 10080=1Week, 43200=1month, )
        self.obs_ticks = 4
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=float(-1.0), high=float(1.0), shape=(self.obs_ticks, 1), dtype=np.float32)
        self.total_reward = 0

    def step(self, action):
        self.prev_actions.append(action)

        self.data_at_step.append(self.my_data[self.tick_count,0])



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
        if self.tick_count == len(self.my_data)-1:
            self.done = True



        self.total_reward += self.reward

        self.all_previous_actions.append(action)
        # create observation:
        info = {"Total_reward": self.total_reward, "tick_count": self.tick_count}
        observation = [self.data_at_step[0], self.data_at_step[1]] + list(self.prev_actions)
        self.observation = np.reshape(np.array(observation),[-1,1])
        self.tick_count +=1
        return self.observation, self.total_reward, self.done, info

    def reset(self):
        # Initial action

        self.prev_actions = deque(maxlen=2)
        self.prev_reward = 0
        self.data_at_step = deque(maxlen=2)
        self.tick_count = 0
        self.done = False
        self.all_previous_actions = []
        #create toy sin data
        # 10k linearly spaced numbers

        self.ticks = np.arange(0, 1000)
        self.my_data = np.reshape(np.sin(np.linspace(-np.pi*60, 60*np.pi, 1000)*100)+1,[-1,1])


        self._first_rendering=True
        for _ in range(2):
            self.prev_actions.append(-1)  # to create history
            self.data_at_step.append(0)
        # create observation:
        observation = [self.data_at_step[0], self.data_at_step[1]] + list(self.prev_actions)
        observation = np.reshape(np.array(observation),[-1,1])

        return observation

    def render(self, mode="human"):
        if self.tick_count < 100:
            return


        def _plot_position():
            color = None

            if self.all_previous_actions[-1]==1:
                color = 'green'
            elif self.all_previous_actions[-1]==2:
                color = 'red'

            if color:
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


        plt.pause(0.01)

    def pause_rendering(self):
        plt.show()