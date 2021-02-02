'''
Implementation of a basic RL environment for continuous spaces.
Includes three test problems which were used in generating the figures.
'''

import numpy as np
import gym
from gym import spaces
import math


#------------------------------------------------------------------------------
'''Sequential Resource Allocation Problem for n agents with K commodities. 
Currently reward is Nash Social Welfare but in the future will integrate more options 
to determine a fair allocation '''

class ResourceAllocationEnvironment(gym.Env):
  """
  Custom Environment that follows gym interface.
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['human']}
  # Define constants for clearer code


  def __init__( self, K = 5, num_agents = 10 , init_budget = 100*np.ones(5), endowments = 100*np.ones(10), type_dist = lambda i: 5*np.random.rand(5), u = lambda x,theta: np.dot(x,theta)):
        '''
        Initializes the Sequential Resource Allocation Environment

        
        K - number of commodities
        num_agents - Number of agents (also the length of an episode)
        init_budget - amount of each commodity the principal begins with
        endowments - how much of the commodities are given to the agents (rough corrolary of "size")
        type_dist: Function determining the types (preferences) of each agent
        u: utility function, given an allocation x and a type theta, u(x,theta) is how good the fit is
        '''
        super(ResourceAllocationEnvironment, self).__init__()

        self.num_commodities = K
        self.epLen = num_agents
        self.budget = init_budget
        self.endowments = endowments
        self.type_dist = lambda i: endowments[i]*type_dist(i)
        self.utility_function = u
        self.starting_state = (init_budget,type_dist(0))
        self.state = self.starting_state
        self.timestep = 0



        # Action space will be choosing K-dimensional allocation vector (vector gets normalized in post but if i could make action space normalize it already)
        self.action_space = spaces.Box(low=0, high=self.budget,
                                        shape=(self.num_commodities,), dtype=np.float32)
        # First K entries of observation space is the remaining budget, last K is the type of the location
        self.observation_space = spaces.Box(low=0, high=self.budget,
                                        shape=(2*self.num_commodities,), dtype=np.float32)

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
        action - vector - chosen action (how much to allocate to prev location)
        Returns:
            reward - double - reward
            newState - int - new state
            done - 0/1 - flag for end of the episode
        '''
        (old_budget,old_type) = self.state
        # new state is sampled from the arrivals distribution

        new_type = self.type_dist(self.timestep)

        # print('old_state' , old_state)
        # print('new_type' , new_type)

        # Uniform Arrivals (have to work out how to use different arrival distributions)
        allocation = np.array(action)
        (new_budget, new_type) = (old_budget-allocation, new_type)

        # print('New state' , newState)
        # Cost is a linear combination of the distance traveled to the action
        # and the distance served to the pickup

        # TODO: INTEGRATE OTHER FAIRNESS METRICS

        reward = np.log(self.utility_function(allocation,old_type))

        # Optionally we can pass additional info, we are not using that for now
        info = {'type' : new_type}

        if self.timestep <= self.epLen:
            pContinue = True
            self.reset()
        else:
            pContinue = False

        self.state = (new_budget, new_type)
        self.action_space = spaces.Box(low=0, high=new_budget,
                                shape=(self.num_commodities,), dtype=np.float32)
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


def make_resource_allocationEnvMDP(K = 5, num_agents = 10 , init_budget = 100*np.ones(5), endowments = 100*np.ones(10), type_dist = lambda i: np.random.rand(5), u = lambda x,theta: np.dot(x,theta)):
    return ResourceAllocationEnvironment(K, num_agents, init_budget, endowments, type_dist,u)
