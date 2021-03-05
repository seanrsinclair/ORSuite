'''
Implementation of a basic RL environment for continuous spaces.
Includes three test problems which were used in generating the figures.
'''

import numpy as np
import gym
from gym import spaces
import math

#------------------------------------------------------------------------------
'''An ambulance environment over [0,1].  An agent interacts through the environment
by picking a location to station the ambulance.  Then a patient arrives and the ambulance
most go and serve the arrival, paying a cost of travel.'''


DEFAULT_CONFIG =  {'epLen': 5, 'arrival_dist': lambda x : np.random.rand(), 'alpha': 0.25, 'starting_state': np.array([0]), 'num_ambulance': 1}

class AmbulanceEnvironment(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the arrivals are always uniformly distributed
  """
  #TODO: Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['human']}


  def __init__(self, config = DEFAULT_CONFIG):
        '''
        For a more detailed description of each parameter, see the readme file
        
        epLen - number of time steps
        arrival_dist - arrival distribution for calls over the space [0,1]
        alpha - parameter for proportional difference in costs
        starting_state - a list containing the starting locations for each ambulance
        num_ambulance - the number of ambulances in the environment 
        '''
        super(AmbulanceEnvironment, self).__init__()

        self.epLen = config['epLen']
        self.alpha = config['alpha']
        self.starting_state = config['starting_state']
        self.state = self.starting_state
        self.timestep = 0
        self.num_ambulance = config['num_ambulance']
        self.arrival_dist = config['arrival_dist']


        # The action space is a box with each ambulances location between 0 and 1
        self.action_space = spaces.Box(low=0, high=1,
                                        shape=(self.num_ambulance,), dtype=np.float32)

        # The observation space is a box with each ambulances location between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(self.num_ambulance,), dtype=np.float32)

  def reset(self):
        """
        TODO: Important: the observation must be a numpy array
        Returns: the starting state
        """
        # Initialize the timestep
        self.timestep = 0
        self.state = self.starting_state

        #TODO: here we convert to float32 to make it more general (in case we want to use continuous actions)
        #TODO: return np.array([self.starting_state]).astype(np.float32)
        return self.starting_state
        #TODO: return a spaces.box object here

  # def arrivals(step):
  #       return np.random.uniform(0,1)

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
        old_state = self.state

        # The location of the new arrival is chosen randomly from the arrivals 
        # distribution arrival_dist
        new_arrival = self.arrival_dist(self.timestep)

        # The closest ambulance to the call is found using the l-1 distance
        close_index = np.argmin(np.abs(old_state - new_arrival))

        # Update the state of the system according to the action taken and change 
        # the location of the closest ambulance to the call to the call location
        newState = np.array(action)
        newState[close_index] = new_arrival
        obs = newState

        # The reward is a linear combination of the distance traveled to the action
        # and the distance traveled to the call
        # alpha controls the tradeoff between cost to travel between arrivals and 
        # cost to travel to a call
        # The reward is negated so that maximizing it will minimize the distance
        reward = -1 * (self.alpha * np.sum(np.abs(self.state - action)) + (1 - self.alpha) * np.max(np.abs(action - newState)))

        # The info dictionary is used to pass the location of the most recent arrival
        # so it can be used by the agent
        info = {'arrival' : new_arrival}

        if self.timestep <= self.epLen:
            pContinue = True
            self.reset()
        else:
            pContinue = False
        self.state = newState
        self.timestep += 1

        #TODO: return self.state and also a box object
        return self.state, reward,  pContinue, info


  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()

  def close(self):
    pass


#-------------------------------------------------------------------------------
# Benchmark environments used when running an experiment


def make_ambulanceEnvMDP(config):
    """ Creates an ambulance line environment with the parameters in config"""
    return AmbulanceEnvironment(config)
