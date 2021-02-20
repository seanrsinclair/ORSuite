import sys


sys.path.append('../src/agent/')
sys.path.append('../src/environment/')
sys.path.append('../src/experiment/')

import numpy as np
import gym

import environment
import experiment
import agent

import resource_allocation 
from equal_allocation_Agent import equalAllocationAgent


import pickle
import time


def run_single_algo(algorithm, algo_information, env, dictionary, path, num_iters, epLen):
    """
    Runs a single instance

    Added new parameter algo_information which is a dictionary mapping string to agent init functions
        These inits assume to only take in epLen as a parameter

    Possible we could make a new file only containing these dictionaries, so we can specify the possible approaches to a problem
    """


    agent_list = [algo_information[algorithm](epLen) for _ in range(num_iters)]
    exp = experiment.Experiment(env, agent_list, dictionary)
    _ = exp.run()
    dt_data = exp.save_data()
    
    # Saving Data
    dt_data.to_csv(path+'.csv')

    return algorithm

''' Defining parameters to be used in the experiment'''


#TODO: Edit algo-list to be the names of the algorithms you created
algo_information = {'Equal_Allocation': lambda epLen: equalAllocationAgent(epLen)}
problem_list = ['default']

for problem in problem_list:
    nEps = 500
    numIters = 15
    #initialize resource allocation environment w/ default parameters
    env = resource_allocation.make_resource_allocationEnvMDP()
    epLen = env.epLen
        ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT


    dictionary = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters, 'epLen': epLen}

    path = {}
    for algorithm in algo_information:
        path[algorithm] = '../data/allocation_%s_%s'%(problem,algorithm)
        run_single_algo(algorithm, algo_information, env, dictionary, path[algorithm], numIters, epLen)
