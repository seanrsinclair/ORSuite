'''
Implementation of an RL environment in a discrete graph space.
'''

import numpy as np
import gym
from gym import spaces
import networkx as nx
import math

from .. import env_configs

#------------------------------------------------------------------------------
'''An ambulance environment over a simple graph.  An agent interacts through 
the environment by [EXPLAIN HOW ENVIRONMENT WORKS HERE] the ambulance.  Then 
a patient arrives and the ambulance most go and serve the arrival, paying a 
cost of travel.'''




class AmbulanceGraphEnvironment(gym.Env):
  """
  Custom Environment that follows gym interface.
  This is a simple env where the arrivals are uniformly distributed across nodes
  """

  metadata = {'render.modes': ['human']}


  def __init__(self, config=env_configs.ambulance_graph_default_config):
        '''
        For a more detailed description of each parameter, see the readme file
        
        epLen - number of time steps
        arrival_dist - arrival distribution for calls over nodes
        alpha - parameter for proportional difference in costs
        edges - edges in the graph and their weights (nodes are automatically inferred)
        starting_state - a list containing the starting nodes for each ambulance
        num_ambulance - the number of ambulances in the environment
        '''
        super(AmbulanceGraphEnvironment, self).__init__()

        self.config = config
        self.epLen = config['epLen']
        self.alpha = config['alpha']
        self.graph = nx.Graph(config['edges'])
        self.starting_state = config['starting_state']
        self.state = self.starting_state
        self.timestep = 0
        self.num_ambulance = config['num_ambulance']
        if config['arrival_dist'] == None:
            num_nodes = len(self.graph.nodes)
            self.arrival_dist = np.full(num_nodes, 1/num_nodes)
        else:
            self.arrival_dist = config['arrival_dist']


        # creates an array stored in space_array the length of the number of ambulances
        # where every entry is the number of nodes in the graph
        num_nodes = self.graph.number_of_nodes()
        space_array = np.full(self.num_ambulance, num_nodes)

        # creates a space where every ambulance can be located at any of the nodes
        self.action_space = spaces.MultiDiscrete(space_array)

        # The definition of the observation space is the same as the action space
        self.observation_space = spaces.MultiDiscrete(space_array)


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
        action - int list - list of nodes the same length as the number of ambulances,
        where each entry i in the list corresponds to the chosen location for 
        ambulance i
        Returns:
            reward - float - reward based on the action chosen
            newState - int list - new state of the system
            done - 0/1 - flag for end of the episode
        '''
        old_state = self.state

        # The location of the new arrival is chosen randomly from among the nodes 
        # in the graph according to the arrival distribution
        new_arrival = np.random.choice(list(self.graph.nodes), p=self.arrival_dist)

        # Finds the distance traveled by all the ambulances from the old state to 
        # the chosen action, assuming that each ambulance takes the shortest path,
        # which is stored in total_dist_oldstate_to_action
        # Also finds the closest ambulance to the call based on their locations at
        # the end of the action, using shortest paths
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

        # Update the state of the system according to the action taken and change 
        # the location of the closest ambulance to the call to the call location
        newState = np.array(action)
        newState[closest_amb_idx] = new_arrival
        obs = newState

        # The reward is a linear combination of the distance traveled to the action
        # and the distance traveled to the call
        # alpha controls the tradeoff between cost to travel between arrivals and 
        # cost to travel to a call
        # The reward is negated so that maximizing it will minimize the distance
        reward = -1 * (self.alpha * total_dist_oldstate_to_action + (1 - self.alpha) * shortest_length)

        # The info dictionary is used to pass the location of the most recent arrival
        # so it can be used by the agent
        info = {'arrival' : new_arrival}

        if self.timestep != (self.epLen-1):
            done = False
        else:
            done = True

        self.state = newState
        self.timestep += 1

        return self.state, reward,  done, info


  def render(self, mode='console'):
    if mode != 'console':
      raise NotImplementedError()

  def close(self):
    pass

