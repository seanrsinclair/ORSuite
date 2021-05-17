import sys

sys.path.append('../')

import numpy as np
import gym
import networkx as nx

import re
import ast
import copy

import or_suite

import os
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd



DEFAULT_CONFIG = or_suite.envs.env_configs.ambulance_graph_default_config
ITHACA_CONFIG = or_suite.envs.env_configs.ambulance_graph_ithaca_config

epLen = DEFAULT_CONFIG['epLen']
nEps = 1000
numIters = 50


def uniform(step, num_nodes):
    return np.array([1 / num_nodes for i in range(num_nodes)])

def nonuniform(step, num_nodes):
    sample = np.random.beta(5,2)
    sample = int(np.floor(num_nodes * sample))
    dist = np.full(num_nodes, 0)
    dist[sample] = 1
    return dist


arrival_dists = [uniform, nonuniform]
num_ambulances = [1, 3]
alphas = [0, 1, 0.25]

environment_config_list = {'default': DEFAULT_CONFIG,
                           'default_ithaca': ITHACA_CONFIG}

for num_ambulance in num_ambulances:
    for alpha in alphas:
        # make a new environment with the data for each alpha and num_ambulance
        CONFIG = copy.deepcopy(ITHACA_CONFIG)
        CONFIG['alpha'] = alpha
        CONFIG['num_ambulance'] = num_ambulance
        CONFIG['starting_state'] = np.array([0 for _ in range(num_ambulance)])

        config_name = "from_data_" + str(alpha) + "_" + str(num_ambulance)
        environment_config_list[config_name] = CONFIG

        for arrival_dist in arrival_dists:
            # make a new environment with each of the other arrival distributions
            print(alpha)
            print(arrival_dist.__name__)

            CONFIG = copy.deepcopy(DEFAULT_CONFIG)
            CONFIG['alpha'] = alpha
            CONFIG['arrival_dist'] = arrival_dist
            CONFIG['num_ambulance'] = num_ambulance
            CONFIG['starting_state'] = np.array([0 for _ in range(num_ambulance)])

            config_name = str(arrival_dist) + "_" + str(alpha) + "_" + str(num_ambulance)
            environment_config_list[config_name] = CONFIG


DEFAULT_SETTINGS = {'seed': 1, 
                    'recFreq': 1, 
                    'dirPath': '../data/ambulance/', 
                    'deBug': False, 
                    'nEps': nEps, 
                    'numIters': numIters, 
                    'saveTrajectory': True, 
                    'epLen' : 5,
                    'render': False,
                    'pickle': False
                    }


for environment in environment_config_list:
    CONFIG = environment_config_list[environment]
    alpha = CONFIG['alpha']
    arrival_dist = CONFIG['arrival_dist']
    num_ambulance = CONFIG['num_ambulance']

    ambulance_graph_env = gym.make('Ambulance-v1', config=CONFIG)
    mon_env = Monitor(ambulance_graph_env)

    agents = {'Random': or_suite.agents.rl.random.randomAgent(), 
            'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']), 
            'Median': or_suite.agents.ambulance.median_graph.medianAgent(CONFIG['epLen'], CONFIG['edges'], CONFIG['num_ambulance']), 
            'Mode': or_suite.agents.ambulance.mode_graph.modeAgent(DEFAULT_CONFIG['epLen']), 
            'SB PPO': PPO(MlpPolicy, mon_env, gamma=1, verbose=0, n_steps=epLen)}

    path_list_line = []
    algo_list_line = []
    path_list_radar = []
    algo_list_radar = []
    for agent in agents:
        print(agent)
        DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_graph_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/'

        if agent == 'SB PPO':
            or_suite.utils.run_single_sb_algo(mon_env, agents[agent], DEFAULT_SETTINGS)
        else:
            or_suite.utils.run_single_algo(ambulance_graph_env, agents[agent], DEFAULT_SETTINGS)

        if num_ambulance > 1 and (agent == 'AdaQL' or agent == 'AdaMB'):
            continue
        path_list_line.append('../data/ambulance_graph_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__))
        algo_list_line.append(str(agent))
        if agent != 'SB PPO':
            path_list_radar.append('../data/ambulance_graph_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__))
            algo_list_radar.append(str(agent))

    fig_path = '../figures/'
    fig_name = 'ambulance_graph'+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'_line_plot'+'.pdf'
    or_suite.plots.plot_line_plots(path_list_line, algo_list_line, fig_path, fig_name, int(nEps / 40)+1)

    additional_metric = {'MRT': lambda traj : or_suite.utils.mean_response_time(traj, lambda x,y: env.lengths[x,y])}
    fig_name = 'ambulance_graph'+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'_radar_plot'+'.pdf'
    or_suite.plots.plot_radar_plots(path_list_radar, algo_list_radar,
    fig_path, fig_name,
    additional_metric
    )

