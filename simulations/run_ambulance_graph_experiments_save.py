import sys


sys.path.append('../or_suite/agents/')
sys.path.append('../or_suite/agents/ambulance/')
sys.path.append('../or_suite/envs/ambulance/')
sys.path.append('../or_suite/experiment/')

import numpy as np
import gym


#import environment
import ambulance_graph

import experiment


import agent

from stable import stableAgent
from median_graph import medianAgent
from mode_graph import modeAgent

import pickle
import time



# def run_single_algo(algorithm, env, dictionary, path, num_iters, epLen, nEps, edges, num_ambulance):

#     agent_list = []

#     for _ in range(num_iters):
#         if algorithm == 'Median':
#             agent_list.append(medianAgent(epLen, edges, num_ambulance))
#         elif algorithm == 'No_Movement':
#             agent_list.append(stableAgent(epLen))
#         elif algorithm == 'Mode':
#             agent_list.append(modeAgent(epLen))


#     # Running Experiments

#     exp = experiment.Experiment(env, agent_list, dictionary)
#     _ = exp.run()
#     dt_data = exp.save_data()


#     # Saving Data
#     dt_data.to_csv(path+'.csv')

#     return algorithm





def run_single_algo(env, agent, settings): 

    exp = experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data(settings[targetPath])


DEFAULT_CONFIG = {'epLen': 5, 'arrival_dist': None, 'alpha': 0.25,
                'edges': [(0,4,{'dist':7}), (0,1,{'dist':1}), (1,2,{'dist':3}), (2,3,{'dist':5}), (1,3,{'dist':1}), (1,4,{'dist':17}), (3,4,{'dist':3})],
                'starting_state': [1,2], 'num_ambulance': 2}

agents = [stableAgent(DEFAULT_CONFIG['epLen']), medianAgent(DEFAULT_CONFIG['epLen'], DEFAULT_CONFIG['edges'], DEFAULT_CONFIG['num_ambulance']), modeAgent(DEFAULT_CONFIG['epLen'])]
nEps = 500
numIters = 50
DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'targetPath': '../data/ambulance_graph/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False}

alphas = [0, 1, 0.25]
arrival_dists = [None, [0.25, 0.4, 0.25, 0.05, 0.05]]


for agent in agents:
    for alpha in alphas:
        for arrival_dist in arrival_dists:
            agent.reset()

            CONFIG = DEFAULT_CONFIG
            CONFIG['alpha'] = alpha
            CONFIG['arrival_dist'] = arrival_dist
            ambulance_graph_env = gym.make('Ambulance-v1', config=CONFIG)

            run_single_algo(ambulance_graph_env, agent, DEFAULT_SETTINGS)







# ''' Defining parameters to be used in the experiment'''

# # ambulance_list = ['shifting', 'beta', 'uniform']
# # ambulance_list = ['uniform']

# arrival_dist_list = [None, [0.25, 0.4, 0.25, 0.05, 0.05]]

# param_list_ambulance = ['0', '1', '25']
# # param_list_ambulance = ['0']


# #TODO: Edit algo-list to be the names of the algorithms you created
# algo_list = ['Median', 'No_Movement', 'Mode']


# #for problem in ambulance_list:
# for arrival_dist in arrival_dist_list:
#     for param in param_list_ambulance:

#         epLen = 5
#         nEps = 500
#         numIters = 50

#         if param == '1':
#             alpha = 1
#         elif param == '0':
#             alpha = 0
#         else:
#             alpha = 0.25

#         edges = [(0,4,{'dist':7}), (0,1,{'dist':1}), (1,2,{'dist':3}), (2,3,{'dist':5}), (1,3,{'dist':1}), (1,4,{'dist':17}), (3,4,{'dist':3})]
#         starting_state = np.asarray([1, 2])
#         num_ambulance = len(starting_state)


#         env = ambulance_graph.make_ambulanceGraphEnvMDP(epLen, arrival_dist, alpha, edges, starting_state, num_ambulance)

#         ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT


#         dictionary = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters, 'epLen': epLen}

#         scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]
#         scaling_list = [0.]

#         path = {}
#         for algorithm in algo_list:
#             path[algorithm] = '../data/ambulance_graph_'+str(arrival_dist)+param+'_'+algorithm
#             run_single_algo(algorithm, env, dictionary, path[algorithm], numIters, epLen, nEps, edges, num_ambulance)
