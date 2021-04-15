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
    A 1-dimensional reinforcement learning environment in the space $X = [0, 1]$. 
    Ambulances are located anywhere in $X = [0,1]$, and at the beginning of each 
    iteration, the agent chooses where to station each ambulance (the action).
    A call arrives, and the nearest ambulance goes to the location of that call.

    Methods: 


    Attributes:

  
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
        self.num_nodes = self.graph.number_of_nodes()
        self.starting_state = config['starting_state']
        self.state = self.starting_state
        self.timestep = 0
        self.num_ambulance = config['num_ambulance']
        self.arrival_dist = config['arrival_dist']

        self.from_data = config['from_data']


        self.lengths = self.find_lengths(self.graph, self.num_nodes)

        if self.from_data:
            self.arrival_data = config['data']
            self.episode_num = 0


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

        if self.from_data:
            self.episode_num += 1


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

        assert self.action_space.contains(action)

        old_state = self.state

        # The location of the new arrival is chosen randomly from among the nodes 
        # in the graph according to the arrival distribution
        prob_list = []
        if self.from_data:
            dataset_step = (self.episode_num * self.epLen + self.timestep) % len(self.arrival_data)
            prob_list = self.arrival_dist(dataset_step, self.num_nodes, self.arrival_data)
        else:
            prob_list = self.arrival_dist(self.timestep, self.num_nodes)
        new_arrival = np.random.choice(self.num_nodes, p=prob_list)

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
            new_length = nx.shortest_path_length(self.graph, action[amb_idx], new_arrival, weight='travel_time')

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

        assert self.observation_space.contains(self.state)

        return self.state, reward,  done, info


    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

    def close(self):
        pass



    def find_lengths(self, graph, num_nodes):
        '''
        Given a graph, find_lengths first calculates the pairwise shortest distance 
        between all the nodes, which is stored in a (symmetric) matrix.
        '''
        dict_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='travel_time'))
        lengths = np.zeros((num_nodes, num_nodes))

        for node1 in range(num_nodes):
            for node2 in range(num_nodes):
                lengths[node1, node2] = dict_lengths[node1][node2]
        return lengths