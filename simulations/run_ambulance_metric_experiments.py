import sys

sys.path.append('../')

import numpy as np
import gym

import or_suite

import copy

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd

import multiprocessing as mp
from joblib import Parallel, delayed



DEFAULT_CONFIG =  or_suite.envs.env_configs.ambulance_metric_default_config
epLen = DEFAULT_CONFIG['epLen']
nEps = 1000
numIters = 50

epsilon = (nEps * epLen)**(-1 / 4)
action_net = np.arange(start=0, stop=1, step=epsilon)
state_net = np.arange(start=0, stop=1, step=epsilon)

scaling_list = [.001, 0.01, 0.1, 0.5, 1., 2.]





DEFAULT_SETTINGS = {'seed': 1, 
                    'recFreq': 1, 
                    'dirPath': '../data/ambulance/', 
                    'deBug': False, 
                    'nEps': nEps, 
                    'numIters': numIters, 
                    'saveTrajectory': True, 
                    'epLen' : 5,
                    'render': False
                    }

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

# arrival_dists = [shifting, uniform, beta]
arrival_dists = [beta]
# num_ambulances = [1,3]
num_ambulances = [1, 3]
# alphas = [0, 0.25, 1]
alphas = [0]

for num_ambulance in num_ambulances:
    for alpha in alphas:
        for arrival_dist in arrival_dists:

            print(alpha)
            print(arrival_dist.__name__)
            CONFIG = copy.deepcopy(DEFAULT_CONFIG)
            CONFIG['alpha'] = alpha
            CONFIG['arrival_dist'] = arrival_dist
            CONFIG['num_ambulance'] = num_ambulance
            CONFIG['starting_state'] = np.array([0 for _ in range(num_ambulance)])
            ambulance_env = gym.make('Ambulance-v0', config=CONFIG)
            mon_env = Monitor(ambulance_env)
            agents = {# 'SB PPO': PPO(MlpPolicy, mon_env, gamma=1, verbose=0, n_steps=epLen),
            'Random': or_suite.agents.rl.random.randomAgent(),
            'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']),
            'Median': or_suite.agents.ambulance.median.medianAgent(DEFAULT_CONFIG['epLen']),
            'AdaQL': or_suite.agents.rl.ada_ql.AdaptiveDiscretizationQL(epLen, scaling_list[0], True, num_ambulance*2),
            # 'AdaMB': or_suite.agents.rl.ada_mb.AdaptiveDiscretizationMB(epLen, scaling_list[0], 0, 2, True, False, num_ambulance, num_ambulance),
            'Unif QL': or_suite.agents.rl.enet_ql.eNetQL(action_net, state_net, epLen, scaling_list[0], (num_ambulance,num_ambulance)),
            'Unif MB': or_suite.agents.rl.enet_mb.eNetMB(action_net, state_net, epLen, scaling_list[0], (num_ambulance,num_ambulance), 0, False)
            }

            path_list_line = []
            algo_list_line = []
            path_list_radar = []
            algo_list_radar= []
            for agent in agents:
                print(agent)
                DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/'
                if agent == 'SB PPO':
                    or_suite.utils.run_single_sb_algo(mon_env, agents[agent], DEFAULT_SETTINGS)
                elif agent == 'AdaQL' or agent == 'Unif QL' or agent == 'AdaMB' or agent == 'Unif MB':
                    or_suite.utils.run_single_algo_tune(ambulance_env, agents[agent], scaling_list, DEFAULT_SETTINGS)
                else:
                    or_suite.utils.run_single_algo(ambulance_env, agents[agent], DEFAULT_SETTINGS)

                path_list_line.append('../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__))
                algo_list_line.append(str(agent))
                if agent != 'SB PPO':
                    path_list_radar.append('../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__))
                    algo_list_radar.append(str(agent))

            fig_path = '../figures/'
            fig_name = 'ambulance_metric'+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'_line_plot'+'.pdf'
            or_suite.plots.plot_line_plots(path_list_line, algo_list_line, fig_path, fig_name, int(nEps / 40)+1)

            additional_metric = {'MRT': lambda traj : or_suite.utils.mean_response_time(traj, lambda x, y : np.abs(x-y))}
            fig_name = 'ambulance_metric'+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'_radar_plot'+'.pdf'
            or_suite.plots.plot_radar_plots(path_list_radar, algo_list_radar,
            fig_path, fig_name,
            additional_metric
            )




