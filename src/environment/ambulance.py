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

class AmbulanceEnvironment(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the arrivals are always uniformly distributed
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['human']}
  # Define constants for clearer code


  def __init__( self, epLen = 5  , arrival_dist = lambda x : np.random.rand() ,alpha = 0.25 , starting_state = np.array([0]) , num_ambulance = 1):
        '''
        epLen - number of steps
        arrivals - arrival distribution for patients
        alpha - parameter for difference in costs
        starting_state - starting location
        '''
        super(AmbulanceEnvironment, self).__init__()

        self.epLen = epLen
        self.alpha = alpha
        self.state = starting_state
        self.starting_state = starting_state
        self.timestep = 0
        self.num_ambulance = num_ambulance
        self.arrival_dist = arrival_dist


        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1,
                                        shape=(self.num_ambulance ,), dtype=np.float32)
        # Example for using a line with (0,10) as observation space:
        self.observation_space = spaces.Box(low=0, high=1,
                                        shape=(self.num_ambulance ,), dtype=np.float32)

  def reset(self):
        """
        Important: the observation must be a numpy array
        :return: (np.array)
        """
        # Initialize the timestep
        self.timestep = 0
        self.state = self.starting_state
        # here we convert to float32 to make it more general (in case we want to use continuous actions)
        # return np.array([self.starting_state]).astype(np.float32)
        return self.starting_state
        ## return a spaces.box object here

  # def arrivals(step):
  #       return np.random.uniform(0,1)

  def step(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action
        Returns:
            reward - double - reward
            newState - int - new state
            done - 0/1 - flag for end of the episode
        '''
        old_state = self.state
        # new state is sampled from the arrivals distribution

        new_arrival = self.arrival_dist(self.timestep)

        # print('old_state' , old_state)
        # print('new_arrival' , new_arrival)

        close_index = np.argmin(np.abs(old_state - new_arrival))
        # print('Close Index' , close_index)

        # Uniform Arrivals (have to work out how to use different arrival distributions)
        newState = np.array(action)
        newState[close_index] = new_arrival
        obs = newState

        # print('New state' , newState)
        # Cost is a linear combination of the distance traveled to the action
        # and the distance served to the pickup

        # TODO: FIX FOR MULTIPLE AMBULANCES

        reward = 1-(self.alpha * np.mean(np.abs(self.state - action)) + (1 - self.alpha) * np.max(np.abs(action - newState)))

        # Optionally we can pass additional info, we are not using that for now
        info = {'arrival' : new_arrival}

        if self.timestep <= self.epLen:
            pContinue = True
            self.reset()
        else:
            pContinue = False
        self.state = newState
        self.timestep += 1

        #return self.state and also a box object
        # can probably return self.state

        return self.state, reward,  pContinue, info

  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()


  def close(self):
    pass


#-------------------------------------------------------------------------------
# Benchmark environments used when running an experiment

#TODO: Make wrapper passing all arguments: e.g.
# def make_ambulanceEnvMDP(epLen = 5  , arrival_dist = lambda x : np.random.rand() ,alpha = 0.25 , starting_state = np.array([0]) , num_ambulance = 1)


def make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state, num_ambulance):
    return AmbulanceEnvironment(epLen, arrivals, alpha, starting_state , num_ambulance)
