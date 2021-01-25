import sys


sys.path.append('../src/agent/')
sys.path.append('../src/environment/')
sys.path.append('../src/experiment/')

import numpy as np
import gym


import environment
import ambulance

import experiment


import agent

from stable_Agent import stableAgent
from median_Agent import medianAgent

import pickle
import time





def run_single_algo(algorithm, env, dictionary, path, num_iters, epLen, nEps):




    agent_list = []

    for _ in range(num_iters):
        if algorithm == 'Median':
            agent_list.append(medianAgent(epLen))
        elif algorithm == 'No_Movement':
            agent_list.append(stableAgent(epLen))


    # Running Experiments

    exp = experiment.Experiment(env, agent_list, dictionary)
    _ = exp.run()
    dt_data = exp.save_data()


    # Saving Data
    dt_data.to_csv(path+'.csv')

    return algorithm

''' Defining parameters to be used in the experiment'''

ambulance_list = ['shifting', 'beta', 'uniform']
# ambulance_list = ['uniform']

param_list_ambulance = ['0', '1', '25']
# param_list_ambulance = ['0']


#TODO: Edit algo-list to be the names of the algorithms you created
algo_list = ['Median', 'No_Movement']


for problem in ambulance_list:
    for param in param_list_ambulance:

        epLen = 5
        nEps = 1000
        numIters = 50
        if problem == 'beta':
            def arrivals(step):
                return np.random.beta(5,2)
        elif problem == 'uniform':
            def arrivals(step):
                return np.random.uniform(0,1)
        elif problem == 'shifting':
            def arrivals(step):
                if step == 0:
                    return np.random.uniform(0, .25)
                elif step == 1:
                    return np.random.uniform(.25, .3)
                elif step == 2:
                    return np.random.uniform(.3, .5)
                elif step == 3:
                    return np.random.uniform(.5, .6)
                else:
                    return np.random.uniform(.6, .65)

        if param == '1':
            alpha = 1
        elif param == '0':
            alpha = 0
        else:
            alpha = 0.25

        starting_state = np.asarray([0.5])


        env = ambulance.make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state, 1)

        ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT


        dictionary = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters, 'epLen': epLen}

        scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]
        scaling_list = [0.]

        path = {}
        for algorithm in algo_list:
            path[algorithm] = '../data/ambulance_'+problem+'_'+param+'_'+algorithm
            run_single_algo(algorithm, env, dictionary, path[algorithm], numIters, epLen, nEps)
