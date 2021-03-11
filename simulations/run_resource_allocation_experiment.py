import sys

sys.path.append('../')

import numpy as np
import gym

import or_suite

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


def run_single_algo(env, agent, settings): 

    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()






''' Defining parameters to be used in the experiment'''


DEFAULT_ENV_CONFIG = {'K':2, 
    'num_rounds':3,
    'weight_matrix':np.array([[1,0],[0,1],[1,1]]),
    'init_budget': 100*np.ones(2),
    'type_dist':lambda i: np.random.randint(50,size=3),
    'utility_function': lambda x,theta: np.dot(x,theta)
    }


#TODO: Edit algo-list to be the names of the algorithms you created
problem_list = ['default']




for problem in problem_list:
    nEps = 500
    numIters = 15
    #initialize resource allocation environment w/ default parameters
    env = or_suite.envs.resource_allocation.resource_allocation.ResourceAllocationEnvironment(config = DEFAULT_ENV_CONFIG)
    epLen = env.epLen
    algo_information = {'Equal_Allocation': or_suite.agents.resource_allocation.equal_allocation.equalAllocationAgent(epLen, DEFAULT_ENV_CONFIG)}

        ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT


    DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance_graph/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : 5}


    path = {}
    for algorithm in algo_information:
        DEFAULT_SETTINGS['dirPath'] = '../data/allocation_%s_%s'%(algorithm,problem)
        run_single_algo(env, algo_information[algorithm], DEFAULT_SETTINGS)
