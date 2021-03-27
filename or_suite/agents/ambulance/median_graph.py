import numpy as np

import networkx as nx
import sklearn_extra.cluster

import sys
from .. import Agent

def find_lengths(graph, num_nodes):
    """
    Given a graph, find_lengths first calculates the pairwise shortest distance 
    between all the nodes, which is stored in a (symmetric) matrix.
    """
    dict_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='travel_time'))
    lengths = np.zeros((num_nodes, num_nodes))

    for node1 in range(num_nodes):
        for node2 in range(num_nodes):
            lengths[node1, node2] = dict_lengths[node1][node2]

    return lengths


''' Agent that implements a median-like heuristic algorithm for the graph ambulance environment'''
class medianAgent(Agent):

    def __init__(self, epLen, edges, num_ambulance):
        '''
        epLen - number of steps
        data - all data observed so far
        edges - the edges and their weights in the environment
        num_ambulance - the number of ambulances in the environment
        avg_inv_lengths - a vector with an entry for each node i that is 
            1/(avg distance between node i and all other nodes)
        call_locs - the node locations of all calls observed so far
        '''
        self.epLen = epLen
        self.data = []
        self.graph = nx.Graph(edges)
        self.num_nodes = self.graph.number_of_nodes()
        self.num_ambulance = num_ambulance
        self.lengths = find_lengths(self.graph, self.num_nodes)
        self.call_locs = []


    def update_config(self, env, config):
        pass

    def reset(self):
        # Resets data and call_locs arrays to be empty
        self.data = []
        self.call_locs = []


    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

        # Adds the most recent state obesrved in the environment to data
        self.data.append(newObs)

        # Adds the most recent arrival location observed to call_locs
        self.call_locs.append(info['arrival'])
        return

    def update_policy(self, k):
        '''Update internal policy based upon records'''

        # Greedy algorithm does not update policy
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''

        # For the first iteration, choose the starting state
        # After that choose locations for ambulances that maximize the number of 
        # calls that arrive at that location multiplied by the inverse of the 
        # average distance between that node and other nodes
        # For a concrete example of how this works, see the ambulance environment
        # readme document
        if timestep == 0:
            return state
        else:
            counts = np.bincount(self.call_locs, minlength=self.num_nodes)
            # print(self.lengths)
            # print(counts)
            score = self.lengths @ counts
            action = []
            for _ in range(self.num_ambulance):
                node = np.argmin(score)
                action.append(node)
                score[node] = 99999999
            return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
