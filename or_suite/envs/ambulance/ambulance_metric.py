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
# import sys
# sys.path.append('..')
from .rendering import PygletWindow, WHITE, RED, GREEN
import time

#------------------------------------------------------------------------------
'''An ambulance environment over [0,1].  An agent interacts through the environment
by picking a location to station the ambulance.  Then a patient arrives and the ambulance
most go and serve the arrival, paying a cost of travel.'''



class AmbulanceEnvironment(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the arrivals are always uniformly distributed
  """

  metadata = {'render.modes': ['human']}


  def __init__(self, config = env_configs.ambulance_metric_default_config):
        '''
        For a more detailed description of each parameter, see the readme file
        
        epLen - number of time steps
        arrival_dist - arrival distribution for calls over the space [0,1]
        alpha - parameter for proportional difference in costs
        starting_state - a list containing the starting locations for each ambulance
        num_ambulance - the number of ambulances in the environment 
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
        # Initialize the timestep
        self.timestep = 0
        self.state = self.starting_state

        return self.starting_state

  def get_config(self):
        return self.config

  def step(self, action):
        '''
        Move one step in the environment

        Args:
        action - float list - list of locations in [0,1] the same length as the 
        number of ambulances, where each entry i in the list corresponds to the 
        chosen location for ambulance i
        Returns:
            reward - float - reward based on the action chosen
            newState - float list - new state
            done - 0/1 - flag for end of the episode
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


  def render(self, mode='human'):
      screen_width = 600
      screen_height = 400
      line_x1 = 50
      line_x2 = 550
      line_y = 300

      if self.viewer is None:
          self.viewer = PygletWindow(screen_width + 50, screen_height + 50)



      if self.most_recent_action is not None:

          self.viewer.reset()

          text = "Current timestep: " + str(self.timestep)
          self.viewer.text(text, line_x1, 0)

          text = "Most recent action"
          self.viewer.text(text, line_x1, 100)

          self.viewer.line(line_x1, line_x2, line_y, width=2, color=WHITE)

          for loc in self.most_recent_action:
              self.viewer.circle(line_x1 + (line_x2 - line_x1) * loc, line_y, radius=5, color=RED)

          self.viewer.update()
          time.sleep(3)




          self.viewer.reset()

          text = "Current timestep: " + str(self.timestep)
          self.viewer.text(text, line_x1, 0)


          text = "Call arrival"
          self.viewer.text(text, line_x1, 100)

          self.viewer.line(line_x1, line_x2, line_y, width=2, color=WHITE)

          for loc in self.most_recent_action:
              self.viewer.circle(line_x1 + (line_x2 - line_x1) * loc, line_y, radius=5, color=RED)

          arrival_loc = self.state[np.argmax(np.abs(self.state - self.most_recent_action))]
          self.viewer.circle(line_x1 + (line_x2 - line_x1) * arrival_loc, line_y, radius=5, color=GREEN)

          self.viewer.update()
          time.sleep(3)



      self.viewer.reset()

      text = "Current timestep: " + str(self.timestep)
      self.viewer.text(text, line_x1, 0)

      text = "Iteration ending state"
      self.viewer.text(text, line_x1, 100)

      self.viewer.line(line_x1, line_x2, line_y, width=2, color=WHITE)

      for loc in self.state:
          self.viewer.circle(line_x1 + (line_x2 - line_x1) * loc, line_y, radius=5, color=RED)

      self.viewer.update()
      time.sleep(3)


  def close(self):
    if self.viewer:
        self.viewer.close()
        self.viewer = None

