'''
Implementation of a basic RL environment for continuous spaces.
Includes three test problems which were used in generating the figures.
'''

import numpy as np
import gym
from gym import spaces
import math
from .. import env_configs
# from gym.envs.classic_control import rendering
# import pyglet
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
renderdir = os.path.dirname(currentdir)
sys.path.append(renderdir)
import rendering 
import time
import pyglet
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)

#------------------------------------------------------------------------------
'''An ambulance environment over [0,1].  An agent interacts through the environment
by picking a location to station the ambulance.  Then a patient arrives and the ambulance
most go and serve the arrival, paying a cost of travel.'''



class AmbulanceEnvironment(gym.Env):
  """
  A 1-dimensional reinforcement learning environment in the space $X = [0, 1]$. 
  Ambulances are located anywhere in $X = [0,1]$, and at the beginning of each 
  iteration, the agent chooses where to station each ambulance (the action).
  A call arrives, and the nearest ambulance goes to the location of that call.

  Methods: 
    reset() : resets the environment to its original settings
    get_config() : returns the config dictionary used to initialize the environment
    step(action) : takes an action from the agent and returns the state of the system after the next arrival
    render(mode) : renders the environment in the mode passed in; 'human' is the only mode currently supported
    close() : closes the window where the rendering is being drawn

  Attributes:
    epLen: (int) number of time steps to run the experiment for
    arrival_dist: (lambda) arrival distribution for calls over the space [0,1]; takes an integer (step) and returns a float between 0 and 1
    alpha: (float) parameter controlling proportional difference in cost to move between calls and to respond to a call
    starting_state: (float list) a list containing the starting locations for each ambulance
    num_ambulance: (int) the number of ambulances in the environment 
    state: (int list) the current state of the environment
    timestep: (int) the timestep the current episode is on
    viewer: (Pyglet window or None) the window where the environment rendering is being drawn
    most_recent_action: (float list or None) the most recent action chosen by the agent (used to render the environment)
    action_space: (Gym.spaces Box) actions must be the length of the number of ambulances, every entry is a float between 0 and 1
    observation_space: (Gym.spaces Box) the environment state must be the length of the number of ambulances, every entry is a float between 0 and 1
  """

  metadata = {'render.modes': ['human']}


  def __init__(self, config = env_configs.ambulance_metric_default_config):
        '''
        Args: 
        config: (dict) a dictionary containing the parameters required to set up a metric ambulance environment
            epLen: (int) number of time steps to run the experiment for
            arrival_dist: (lambda) arrival distribution for calls over the space [0,1]; takes an integer (step) and returns a float between 0 and 1
            alpha: (float) parameter controlling proportional difference in cost to move between calls and to respond to a call
            starting_state: (float list) a list containing the starting locations for each ambulance
            num_ambulance: (int) the number of ambulances in the environment 
        '''
        super(AmbulanceEnvironment, self).__init__()

        self.config = config
        self.epLen = config['epLen']
        self.alpha = config['alpha']
        self.starting_state = config['starting_state']
        self.state = np.array(self.starting_state, dtype=np.float32)
        self.timestep = 0
        self.num_ambulance = config['num_ambulance']
        self.arrival_dist = config['arrival_dist']

        # variables used for rendering code
        self.viewer = None
        self.most_recent_action = None

        # The action space is a box with each ambulances location between 0 and 1
        self.action_space = spaces.Box(low=0, high=1,
                                        shape=(self.num_ambulance,), dtype=np.float32)

        # The observation space is a box with each ambulances location between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(self.num_ambulance,), dtype=np.float32)

  def reset(self):
        """
        Reinitializes variables and returns the starting state
        """

        self.timestep = 0
        self.state = self.starting_state

        return self.starting_state

  def get_config(self):
        return self.config

  def step(self, action):
        '''
        Move one step in the environment

        Args:
            action: (float list) list of locations in [0,1] the same length as the 
            number of ambulances, where each entry i in the list corresponds to the 
            chosen location for ambulance i
        Returns:
            reward: (float) reward based on the action chosen
            newState: (float list) the state of the environment after the action and call arrival
            done: (bool) flag indicating the end of the episode
        '''

        assert self.action_space.contains(action)

        old_state = np.array(self.state)

        # The location of the new arrival is chosen randomly from the arrivals 
        # distribution arrival_dist
        new_arrival = self.arrival_dist(self.timestep)

        # The closest ambulance to the call is found using the l-1 distance
        close_index = np.argmin(np.abs(old_state - new_arrival))

        # Update the state of the system according to the action taken and change 
        # the location of the closest ambulance to the call to the call location
        action = np.array(action, dtype=np.float32)
        self.most_recent_action = action
        new_state = action.copy()
        new_state[close_index] = new_arrival

        # print("Old", old_state)
        # print("Action", action)
        # print("Close Index", close_index)
        # print("New Arrival", new_arrival)
        # print("New", new_state)

        # The reward is a linear combination of the distance traveled to the action
        # and the distance traveled to the call
        # alpha controls the tradeoff between cost to travel between arrivals and 
        # cost to travel to a call
        # The reward is negated so that maximizing it will minimize the distance

        # print("alpha", self.alpha)

        reward = -1 * (self.alpha * np.sum(np.abs(old_state - action)) + (1 - self.alpha) * np.sum(np.abs(action - new_state)))
        
        
        # The info dictionary is used to pass the location of the most recent arrival
        # so it can be used by the agent
        info = {'arrival' : new_arrival}

        if self.timestep != self.epLen - 1:
            done = False
        else:
            done = True

        self.state = new_state
        self.timestep += 1

        assert self.observation_space.contains(self.state)

        return self.state, reward,  done, info


  def reset_current_step(self, text, line_x1, line_x2, line_y):
      # Used to render a textbox saying the current timestep
      self.viewer.reset()
      self.viewer.text("Current timestep: " + str(self.timestep), line_x1, 0)
      self.viewer.text(text, line_x1, 100)
      self.viewer.line(line_x1, line_x2, line_y, width=2, color=rendering.WHITE)

  def draw_ambulances(self, locations, line_x1, line_x2, line_y, ambulance):
      for loc in locations:
            self.viewer.image(line_x1 + (line_x2 - line_x1) * loc, line_y, ambulance, 0.15)
            # self.viewer.circle(line_x1 + (line_x2 - line_x1) * loc, line_y, radius=5, color=rendering.RED)


  def render(self, mode='human'):
      # Renders the environment using a pyglet window
      screen_width = 600
      screen_height = 400
      line_x1 = 50
      line_x2 = 550
      line_y = 300

      ambulance = pyglet.image.load('images/ambulance.jpg')
      call = pyglet.image.load('images/call.jpg')

      if self.viewer is None:
          self.viewer = rendering.PygletWindow(screen_width + 50, screen_height + 50)


      if self.most_recent_action is not None:

          self.reset_current_step("Action chosen", line_x1, line_x2, line_y)
          self.draw_ambulances(self.most_recent_action, line_x1, line_x2, line_y, ambulance)
          self.viewer.update()
          time.sleep(2)


          self.reset_current_step("Call arrival", line_x1, line_x2, line_y)
          self.draw_ambulances(self.most_recent_action, line_x1, line_x2, line_y, ambulance)
          
          arrival_loc = self.state[np.argmax(np.abs(self.state - self.most_recent_action))]
          self.viewer.image(line_x1 + (line_x2 - line_x1) * arrival_loc, line_y, call, 0.05)
        #   self.viewer.circle(line_x1 + (line_x2 - line_x1) * arrival_loc, line_y, radius=5, color=rendering.GREEN)
          self.viewer.update()
          time.sleep(2)


      self.reset_current_step("Iteration ending state", line_x1, line_x2, line_y)

      self.draw_ambulances(self.state, line_x1, line_x2, line_y, ambulance)

      self.viewer.update()
      time.sleep(2)


  def close(self):
    # Closes the rendering window
    if self.viewer:
        self.viewer.close()
        self.viewer = None

