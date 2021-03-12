import sys

<<<<<<< HEAD


=======
>>>>>>> 72b65ac (Line figures and plots)
sys.path.append('../')

import numpy as np
import gym

import or_suite

from stable_baselines3 import PPO
<<<<<<< HEAD
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env

=======
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
>>>>>>> 72b65ac (Line figures and plots)


def run_single_algo(env, agent, settings): 

    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()


<<<<<<< HEAD
''' Defining parameters to be used in the experiment'''


DEFAULT_ENV_CONFIG = or_suite.envs.env_configs.resource_allocation_default_cofig


# #TODO: Edit algo-list to be the names of the algorithms you created
problem_list = ['default']


=======




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




>>>>>>> 72b65ac (Line figures and plots)
for problem in problem_list:
    nEps = 500
    numIters = 15
    #initialize resource allocation environment w/ default parameters
<<<<<<< HEAD
    env = gym.make('Resource-v0', config = DEFAULT_ENV_CONFIG)
    epLen = env.epLen
    algo_information = {'Random': or_suite.agents.rl.random.randomAgent(), 'Equal_Allocation': or_suite.agents.resource_allocation.equal_allocation.equalAllocationAgent(epLen, DEFAULT_ENV_CONFIG)}
=======
    env = or_suite.envs.resource_allocation.resource_allocation.ResourceAllocationEnvironment(config = DEFAULT_ENV_CONFIG)
    epLen = env.epLen
    algo_information = {'Equal_Allocation': or_suite.agents.resource_allocation.equal_allocation.equalAllocationAgent(epLen, DEFAULT_ENV_CONFIG)}
>>>>>>> 72b65ac (Line figures and plots)

        ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT


<<<<<<< HEAD
    DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance_graph/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : epLen}
=======
    DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance_graph/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : 5}
>>>>>>> 72b65ac (Line figures and plots)


    path = {}
    for algorithm in algo_information:
        DEFAULT_SETTINGS['dirPath'] = '../data/allocation_%s_%s'%(algorithm,problem)
        run_single_algo(env, algo_information[algorithm], DEFAULT_SETTINGS)
<<<<<<< HEAD

# env = gym.make('Resource-v0', config = DEFAULT_ENV_CONFIG)


# check_env(env)

# model = PPO(CnnPolicy, env, verbose=1, gamma=1)
# model.learn(total_timesteps=1000)

# env = gym.make('Resource-v0')
# n_episodes = 100
# res_mean, res_std = evaluate_policy(model, env, n_eval_episodes=n_episodes)

# print(-res_mean, '+/-', 1.96*res_std/np.sqrt(n_episodes))
=======
>>>>>>> 72b65ac (Line figures and plots)
