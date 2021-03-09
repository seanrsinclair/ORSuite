import sys

sys.path.append('../')


import numpy as np
import gym

import or_suite

def run_single_algo(env, agent, settings): 

    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()




DEFAULT_CONFIG = {'epLen': 5, 'arrival_dist': None, 'alpha': 0.25,
                'edges': [(0,4,{'dist':7}), (0,1,{'dist':1}), (1,2,{'dist':3}), (2,3,{'dist':5}), (1,3,{'dist':1}), (1,4,{'dist':17}), (3,4,{'dist':3})],
                'starting_state': [1,2], 'num_ambulance': 2}



# agents = [or_suite.agents.ambulance.stableAgent(DEFAULT_CONFIG['epLen']), or_suite.agents.ambulance.medianAgent(DEFAULT_CONFIG['epLen'], DEFAULT_CONFIG['edges'], DEFAULT_CONFIG['num_ambulance']), modeAgent(DEFAULT_CONFIG['epLen'])]
agents = [or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen'])]

nEps = 500
numIters = 50
DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance_graph/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : 5}

alphas = [0, 1, 0.25]
arrival_dists = [None, [0.25, 0.4, 0.25, 0.05, 0.05]]


for agent in agents:
    for alpha in alphas:
        for arrival_dist in arrival_dists:
            agent.reset()

            CONFIG = DEFAULT_CONFIG
            CONFIG['alpha'] = alpha
            CONFIG['arrival_dist'] = arrival_dist
            DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_graph_'+str(alpha)+'_'+str(arrival_dist)+'/'
            ambulance_graph_env = gym.make('Ambulance-v1', config=CONFIG)

            run_single_algo(ambulance_graph_env, agent, DEFAULT_SETTINGS)




