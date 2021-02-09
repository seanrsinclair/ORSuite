import numpy as np
import agent
import networkx as nx
import sklearn_extra.cluster




def find_median(graph):
    """
    for node in graph
        add up distances to other nodes
        choose node w smallest distance and remove from graph
    """
    median_node = 0
    median_node_dist = 99999999
    for node1 in graph.nodes:
        total_dist = 0

        for node2 in graph.nodes:
            total_dist += nx.shortest_path_length(graph, node1, node2, weight='dist')

        if total_dist < median_node_dist:
            median_node = node1
            median_node_dist = total_dist

    graph.remove_node(median_node)
    return median_node, graph


''' Agent which implements several heuristic algorithms'''
class medianAgent(agent.FiniteHorizonAgent):

    def __init__(self, epLen, edges, num_ambulance):
        '''args:
            epLen - number of steps
            func - function used to decide action
            data - all data observed so far
            alpha - alpha parameter in ambulance problem
        '''
        self.epLen = epLen
        self.data = []
        self.graph = nx.Graph(edges)
        self.num_ambulance = num_ambulance

        self.median_nodes = []
        for amb_idx in range(num_ambulance):
            median, new_graph = find_median(self.graph)
            self.median_nodes.append(median)
            self.graph = new_graph


    def reset(self):
        # resets data matrix to be empty
        self.data = []
        self.graph = nx.Graph(edges)
        self.median_nodes = []

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        self.data.append(newObs)
        #self.call_locs.append(info['arrival'])
        return

    def get_num_arms(self):
        return 0

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''
        return self.median_nodes


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
