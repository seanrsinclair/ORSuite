
import sys

sys.path.append('../')

import numpy as np
import gym
import pickle

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



DEFAULT_CONFIG =  or_suite.envs.env_configs.oil_environment_default_config
epLen = DEFAULT_CONFIG['epLen']
nEps = 50
numIters = 10

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

def laplace(x,a,h, lam):
    return np.exp((-1)*lam*np.sum(np.abs(x-0.11*h)))

# probs = [shifting, uniform, beta]
prob_list = [laplace]
# dims = [1,3]
dim_list = [1]
# cost_params = [0, 0.25, 1]
cost_params = [0]

for dim in dim_list:
    for cost_param in cost_params:
        for prob in prob_list:

            print(cost_param)
            print(prob.__name__)
            CONFIG = copy.deepcopy(DEFAULT_CONFIG)
            CONFIG['cost_param'] = cost_param
            CONFIG['oil_prob'] = lambda x,a,h: prob(x,a,h,1)
            CONFIG['dim'] = dim
            CONFIG['starting_state'] = np.array([0 for _ in range(dim)])
            oil_env = gym.make('Oil-v0', config=CONFIG)
            mon_env = Monitor(oil_env)
            agents = { 'SB PPO': PPO(MlpPolicy, mon_env, gamma=1, verbose=0, n_steps=epLen),
            'Random': or_suite.agents.rl.random.randomAgent(),
            'AdaQL': or_suite.agents.rl.ada_ql.AdaptiveDiscretizationQL(epLen, scaling_list[0], True, dim*2),
            'AdaMB': or_suite.agents.rl.ada_mb.AdaptiveDiscretizationMB(epLen, scaling_list[0], 0, 2, True, True, dim, dim),
            'Unif QL': or_suite.agents.rl.enet_ql.eNetQL(action_net, state_net, epLen, scaling_list[0], (dim,dim)),
            'Unif MB': or_suite.agents.rl.enet_mb.eNetMB(action_net, state_net, epLen, scaling_list[0], (dim,dim), 0, False),
            }

            path_list_line = []
            algo_list_line = []
            path_list_radar = []
            algo_list_radar= []
            for agent in agents:
                print(agent)
                DEFAULT_SETTINGS['dirPath'] = '../data/oil_metric_'+str(agent)+'_'+str(dim)+'_'+str(cost_param)+'_'+str(prob.__name__)+'/'
                if agent == 'SB PPO':
                    or_suite.utils.run_single_sb_algo(mon_env, agents[agent], DEFAULT_SETTINGS)
                elif agent == 'AdaQL' or agent == 'Unif QL' or agent == 'AdaMB' or agent == 'Unif MB':
                    or_suite.utils.run_single_algo_tune(oil_env, agents[agent], scaling_list, DEFAULT_SETTINGS)
                else:
                    or_suite.utils.run_single_algo(oil_env, agents[agent], DEFAULT_SETTINGS)

                path_list_line.append('../data/oil_metric_'+str(agent)+'_'+str(dim)+'_'+str(cost_param)+'_'+str(prob.__name__))
                algo_list_line.append(str(agent))
                if agent != 'SB PPO':
                    path_list_radar.append('../data/oil_metric_'+str(agent)+'_'+str(dim)+'_'+str(cost_param)+'_'+str(prob.__name__))
                    algo_list_radar.append(str(agent))
                file_name = '../data/oil_metric_'+str(agent)+'_'+str(dim)+'_'+str(cost_param)+'_'+str(prob.__name__)+'/agent.obj'
                outfile = open(file_name, 'wb')
                pickle.dump(agent, outfile)
                outfile.close()

            fig_path = '../figures/'
            fig_name = 'oil_metric'+'_'+str(dim)+'_'+str(cost_param)+'_'+str(prob.__name__)+'_line_plot'+'.pdf'
            or_suite.plots.plot_line_plots(path_list_line, algo_list_line, fig_path, fig_name, int(nEps / 40)+1)

            additional_metric = {}
            fig_name = 'oil_metric'+'_'+str(dim)+'_'+str(cost_param)+'_'+str(prob.__name__)+'_radar_plot'+'.pdf'
            or_suite.plots.plot_radar_plots(path_list_radar, algo_list_radar,
            fig_path, fig_name,
            additional_metric
            )




