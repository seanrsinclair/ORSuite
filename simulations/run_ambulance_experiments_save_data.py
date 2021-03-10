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


DEFAULT_CONFIG =  {'epLen': 5, 'arrival_dist': lambda x : np.random.rand(), 'alpha': 0.25, 
                    'starting_state': np.array([0]), 'num_ambulance': 1}


agents = [or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']), or_suite.agents.ambulance.median.medianAgent(DEFAULT_CONFIG['epLen'])]
nEps = 50
numIters = 20
epLen = 5
DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance_metric/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen': epLen}

alphas = [0, 1, 0.25]

def shifting(step):
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

def uniform(step):
    return np.random.uniform(0,1)

def beta(step):
    return np.random.beta(5,2)

arrival_dists = [shifting, uniform, beta]


for agent in agents:
    for alpha in alphas:
        for arrival_dist in arrival_dists:
            agent.reset()

            CONFIG = DEFAULT_CONFIG
            CONFIG['alpha'] = alpha
            CONFIG['arrival_dist'] = arrival_dist
            ambulance_graph_env = gym.make('Ambulance-v0', config=CONFIG)

            run_single_algo(ambulance_graph_env, agent, DEFAULT_SETTINGS)



######## Testing with Stable Baselines3 PPO Algorithm ########

env = make_vec_env('Ambulance-v1', n_envs=4)
model = PPO(MlpPolicy, env, verbose=1, gamma=1)
model.learn(total_timesteps=1000)

env = gym.make('Ambulance-v1')
n_episodes = 100
res_mean, res_std = evaluate_policy(model, env, n_eval_episodes=n_episodes)

print(-res_mean, '+/-', 1.96*res_std/np.sqrt(n_episodes))



















# ambulance_list = ['shifting', 'beta', 'uniform']
# # ambulance_list = ['uniform']

# param_list_ambulance = ['0', '1', '25']
# # param_list_ambulance = ['0']


# #TODO: Edit algo-list to be the names of the algorithms you created
# algo_list = ['Median', 'No_Movement']


# for problem in ambulance_list:
#     for param in param_list_ambulance:

#         epLen = 5
#         nEps = 500
#         numIters = 15
#         if problem == 'beta':
#             def arrivals(step):
#                 return np.random.beta(5,2)
#         elif problem == 'uniform':
#             def arrivals(step):
#                 return np.random.uniform(0,1)
#         elif problem == 'shifting':
#             def arrivals(step):
#                 if step == 0:
#                     return np.random.uniform(0, .25)
#                 elif step == 1:
#                     return np.random.uniform(.25, .3)
#                 elif step == 2:
#                     return np.random.uniform(.3, .5)
#                 elif step == 3:
#                     return np.random.uniform(.5, .6)
#                 else:
#                     return np.random.uniform(.6, .65)

#         if param == '1':
#             alpha = 1
#         elif param == '0':
#             alpha = 0
#         else:
#             alpha = 0.25

#         starting_state = np.asarray([0.5])


#         env = ambulance.make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state, 1)

#         ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT


#         dictionary = {'seed': 1, 'epFreq' : 1, 'targetPath': './tmp.csv', 'deBug' : False, 'nEps': nEps, 'recFreq' : 10, 'numIters' : numIters, 'epLen': epLen}

#         scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]
#         scaling_list = [0.]

#         path = {}
#         for algorithm in algo_list:
#             path[algorithm] = '../data/ambulance_'+problem+'_'+param+'_'+algorithm
#             run_single_algo(algorithm, env, dictionary, path[algorithm], numIters, epLen, nEps)
