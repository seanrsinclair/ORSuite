import numpy as np
import agent
import networkx as nx
import sklearn_extra.cluster



def find_lengths(graph):
    nodes = list(graph.nodes)
    num_nodes = len(nodes)
    dict_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, cutoff=None, weight='dist'))
    lengths = np.zeros((num_nodes, num_nodes))

    for node1 in nodes:
        for node2 in nodes:
            lengths[node1, node2] = dict_lengths[node1][node2]
    
    avg_inv_lengths = 1 / np.mean(lengths, axis=0)

    return avg_inv_lengths


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
        self.num_nodes = len(self.graph.nodes)
        self.num_ambulance = num_ambulance
        self.avg_inv_lengths = find_lengths(self.graph)
        self.call_locs = []


    def reset(self):
        # resets data matrix to be empty
        self.data = []
        self.graph = nx.Graph(edges)
        self.call_locs = []


    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        self.data.append(newObs)
        self.call_locs.append(info['arrival'])
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
        if timestep == 0:
            return state
        else:
            num_ambulance = len(self.data[0])
            counts = np.bincount(self.call_locs, minlength=self.num_nodes)
            score = np.multiply(self.avg_inv_lengths, counts)
            action = []
            for i in range(num_ambulance):
                node = np.argmax(score)
                action.append(node)
                counts[node] = 0
            return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
