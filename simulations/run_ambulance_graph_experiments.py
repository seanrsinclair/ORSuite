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


def run_single_algo(env, agent, settings): 

    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    exp.run()
    exp.save_data()


DEFAULT_CONFIG = or_suite.envs.env_configs.ambulance_graph_default_config

agents = {'Random': or_suite.agents.rl.random.randomAgent(), 'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']), 'Median': or_suite.agents.ambulance.median_graph.medianAgent(DEFAULT_CONFIG['epLen'], DEFAULT_CONFIG['edges'], DEFAULT_CONFIG['num_ambulance']), 'Mode': or_suite.agents.ambulance.mode_graph.modeAgent(DEFAULT_CONFIG['epLen']), 'SB_PPO': None}
# agents = {'Random': or_suite.agents.rl.random.randomAgent(), 'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']), 'Median': or_suite.agents.ambulance.median_graph.medianAgent(DEFAULT_CONFIG['epLen'], DEFAULT_CONFIG['edges'], DEFAULT_CONFIG['num_ambulance']), 'Mode': or_suite.agents.ambulance.mode_graph.modeAgent(DEFAULT_CONFIG['epLen'])}

# agents = {'SB_PPO': None}
nEps = 10
numIters = 50
epLen = DEFAULT_CONFIG['epLen']
DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : 5}

alphas = [0, 1, 0.25]
num_ambulances = [1,3]
# num_ambulances = [3]

def uniform(step, num_nodes):
    return np.array([1 / num_nodes for i in range(num_nodes)])

def nonuniform(step, num_nodes):
    #TODO: fix so it actually works for a variable number of nodes
    return np.array([0.25, 0.4, 0.25, 0.05, 0.05])


edges_file = open("ithaca.edgelist", "r")
ithaca_edges = []
for line in edges_file:
    travel_dict = ast.literal_eval(re.search('({.+})', line).group(0))
    split = line.split()
    ithaca_edges.append((int(split[0]), int(split[1]), travel_dict))
edges_file.close()


arrivals_file = open("arrivals.txt", "r")
ithaca_arrivals = arrivals_file.read().splitlines()
ithaca_arrivals = [int(i) for i in ithaca_arrivals]
arrivals_file.close()

def from_data(step, num_nodes, ithaca_arrivals):
    node = ithaca_arrivals[step]
    dist = np.full(num_nodes, 0)
    dist[node] = 1
    return dist

arrival_dists = [uniform, nonuniform, from_data]


for agent in agents:
    for num_ambulance in num_ambulances:
        for alpha in alphas:
            for arrival_dist in arrival_dists:
                print(agent)
                print(num_ambulance)
                print(alpha)
                print(arrival_dist.__name__)
                CONFIG = copy.deepcopy(DEFAULT_CONFIG)
                CONFIG['alpha'] = alpha
                CONFIG['arrival_dist'] = arrival_dist
                CONFIG['num_ambulance'] = num_ambulance
                CONFIG['starting_state'] = [0 for _ in range(num_ambulance)]

                if arrival_dist == from_data:
                    CONFIG['from_data'] = True
                    CONFIG['edges'] = ithaca_edges
                    CONFIG['data'] = ithaca_arrivals

                DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_graph_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/'
                ambulance_graph_env = gym.make('Ambulance-v1', config=CONFIG)
                if agent == 'SB_PPO':
                    episodes = []
                    iterations = []
                    rewards = []
                    times = []
                    memory = []
                
                    for i in range(numIters):
                        sb_env = Monitor(ambulance_graph_env)
                        model = PPO(MlpPolicy, sb_env, gamma=1, n_steps = epLen)
                        model.learn(total_timesteps=epLen*nEps)

                        episodes = np.append(episodes,np.arange(0, nEps))
                        iterations = np.append(iterations, [i for _ in range(nEps)])
                        rewards =np.append(rewards, sb_env.get_episode_rewards())
                        times = np.append(times, sb_env.get_episode_times())
                        memory = np.append(memory, np.zeros(len(sb_env.get_episode_rewards())))

                    df = pd.DataFrame({'episode': episodes,
                            'iteration': iterations,
                            'epReward': rewards,
                            'time': times,
                            'memory': memory})
                    
                    if not os.path.exists(DEFAULT_SETTINGS['dirPath']):
                        os.makedirs(DEFAULT_SETTINGS['dirPath'])
                    df.to_csv(DEFAULT_SETTINGS['dirPath']+'data.csv', index=False, float_format='%.2f', mode='w')
                else:
                    if agent == 'Median':
                        agent_to_use = or_suite.agents.ambulance.median_graph.medianAgent(CONFIG['epLen'], CONFIG['edges'], CONFIG['num_ambulance'])
                    else:
                        agent_to_use = agents[agent]
                    run_single_algo(ambulance_graph_env, agent_to_use, DEFAULT_SETTINGS)


for alpha in alphas:
    for arrival_dist in arrival_dists:
        path_list_line = []
        algo_list_line = []

        path_list_radar = []
        algo_list_radar = []
        for agent in agents:
            path_list_line.append('../data/ambulance_graph_'+str(agent)+'_'+str(alpha)+'_'+str(arrival_dist)+'/data.csv')
            algo_list_line.append(str(agent))
            if agent != 'SB_PPO':
                path_list_radar.append('../data/ambulance_graph_'+str(agent)+'_'+str(alpha)+'_'+str(arrival_dist)+'/data.csv')
                algo_list_radar.append(str(agent))

        fig_path = '../figures/'
        fig_name = 'ambulance_graph_'+str(alpha)+'_'+str(arrival_dist)+'_line_plot'+'.pdf'
        or_suite.plots.plot_line_plots(path_list_line, algo_list_line, fig_path, fig_name)

        fig_name = 'ambulance_graph_'+str(alpha)+'_'+str(arrival_dist)+'_radar_plot'+'.pdf'
        or_suite.plots.plot_radar_plots(path_list_radar, algo_list_radar, fig_path, fig_name)
        path_list_line = []
        algo_list_line = []

        path_list_radar = []
        algo_list_radar = []
        for agent in agents:
            path_list_line.append('../data/ambulance_graph_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/data.csv')
            algo_list_line.append(str(agent))
            if agent != 'SB_PPO':
                path_list_radar.append('../data/ambulance_graph_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/data.csv')
                algo_list_radar.append(str(agent))

        fig_path = '../figures/'
        fig_name = 'ambulance_graph_'+str(num_ambulance) + '_'+ str(alpha)+'_'+str(arrival_dist.__name__)+'_line_plot'+'.pdf'
        or_suite.plots.plot_line_plots(path_list_line, algo_list_line, fig_path, fig_name)

        fig_name = 'ambulance_graph_'+str(num_ambulance) + '_' + str(alpha)+'_'+str(arrival_dist.__name__)+'_radar_plot'+'.pdf'
        or_suite.plots.plot_radar_plots(path_list_radar, algo_list_radar, fig_path, fig_name)


