'''
Implementation of an RL environment in a discrete graph space.
'''

import numpy as np
import gym
#from gym import spaces
import networkx as nx
import math

#------------------------------------------------------------------------------
'''An ambulance environment over a simple graph.  An agent interacts through 
the environment by [EXPLAIN HOW ENVIRONMENT WORKS HERE] the ambulance.  Then 
a patient arrives and the ambulance most go and serve the arrival, paying a 
cost of travel.'''

class AmbulanceEnvironment(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the arrivals are uniformly distributed across nodes
  """
  # Because of google colab, we cannot implement the GUI ('human' render mode)
  metadata = {'render.modes': ['human']}
  # Define constants for clearer code


  def __init__(self, epLen = 5, arrival_dist = lambda x : np.random.rand(), alpha = 0.25,
                edges = [(1,2,{'dist':3}), (2,3,{'dist':5}), (1,3,{'dist':1})], num_ambulance = 1, starting_state = [2]):
        '''
        epLen - number of steps
        arrivals - arrival distribution for patients
        alpha - parameter for difference in costs
        starting_state - starting location
        '''
        super(AmbulanceEnvironment, self).__init__()

        self.epLen = epLen
        self.alpha = alpha

        self.graph = nx.Graph(edges)

        self.state = starting_state
        self.starting_state = starting_state
        self.timestep = 0
        self.num_ambulance = num_ambulance
        self.arrival_dist = arrival_dist


        # Example when using discrete actions:
        # self.action_space = spaces.Box(low=0, high=1,
        #                               ac  shape=(self.num_ambulance ,), dtype=np.float32)
        # Example for using a line with (0,10) as observation space:
        # self.observation_space = spaces.Box(low=0, high=1,
        #                                 shape=(self.num_ambulance ,), dtype=np.float32)

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

        # chooses randomly from among all the nodes in the graph

        # TODO: CHANGE THIS TO ACTUALLY USE ARRIVAL_DIST
        new_arrival = np.random.choice(list(self.graph.nodes))

        # print('old_state' , old_state)
        # print('new_arrival' , new_arrival)


        shortest_length = 999999999
        closest_amb_idx = 0
        closest_amb_loc = action[closest_amb_idx]

        total_dist_oldstate_to_action = 0

        for amb_idx in range(len(action)):
            new_length = nx.shortest_path_length(self.graph, action[amb_idx], new_arrival, weight='dist')

            total_dist_oldstate_to_action += nx.shortest_path_length(self.graph, self.state[amb_idx], action[amb_idx], weight='dist')

            if new_length < shortest_length:
                shortest_length = new_length
                closest_amb_idx = amb_idx
                closest_amb_loc = action[closest_amb_idx]
            else:
                continue

        # print('Close Index' , close_index)

        # Uniform Arrivals (have to work out how to use different arrival distributions)
        newState = np.array(action)
        newState[closest_amb_idx] = new_arrival
        obs = newState

        # print('New state' , newState)
        # Cost is a linear combination of the distance traveled to the action
        # and the distance served to the pickup

        reward = -1 * (self.alpha * total_dist_oldstate_to_action + (1 - self.alpha) * shortest_length)

        # Optionally we can pass additional info, we are not using that for now
        info = {'arrival' : new_arrival}

        if self.timestep <= self.epLen:
            pContinue = True

            ## TODO: why do we do self.reset() every time?
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
