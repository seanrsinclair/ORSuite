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


DEFAULT_CONFIG = or_suite.envs.env_configs.ambulance_graph_default_config

agents = {'Random': or_suite.agents.rl.random.randomAgent(), 'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']), 'Median': or_suite.agents.ambulance.median_graph.medianAgent(DEFAULT_CONFIG['epLen'], DEFAULT_CONFIG['edges'], DEFAULT_CONFIG['num_ambulance']), 'Mode': or_suite.agents.ambulance.mode_graph.modeAgent(DEFAULT_CONFIG['epLen'])}
nEps = 10
numIters = 5
epLen = DEFAULT_CONFIG['epLen']
DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : 5}

alphas = [0, 1, 0.25]
arrival_dists = [None, [0.25, 0.4, 0.25, 0.05, 0.05]]


# for agent in agents:
#     for alpha in alphas:
#         for arrival_dist in arrival_dists:
#             CONFIG = DEFAULT_CONFIG
#             CONFIG['alpha'] = alpha
#             CONFIG['arrival_dist'] = arrival_dist
#             DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_graph_'+str(agent)+'_'+str(alpha)+'_'+str(arrival_dist)+'/'
#             ambulance_graph_env = gym.make('Ambulance-v1', config=CONFIG)

#             run_single_algo(ambulance_graph_env, agents[agent], DEFAULT_SETTINGS)

# for alpha in alphas:
#     for arrival_dist in arrival_dists:
#         path_list = []
#         algo_list = []
#         for agent in agents:
#             path_list.append('../data/ambulance_graph_'+str(agent)+'_'+str(alpha)+'_'+str(arrival_dist)+'/data.csv')
#             algo_list.append(str(agent))

#         fig_path = '../figures/'
#         fig_name = 'ambulance_graph_'+str(alpha)+'_'+str(arrival_dist)+'_line_plot'+'.pdf'
#         or_suite.plots.plot_line_plots(path_list, algo_list, fig_path, fig_name)

#         fig_name = 'ambulance_graph_'+str(alpha)+'_'+str(arrival_dist)+'_radar_plot'+'.pdf'
#         or_suite.plots.plot_radar_plots(path_list, algo_list, fig_path, fig_name)


######## Testing with Stable Baselines3 PPO Algorithm ########

env = gym.make('Ambulance-v1')





model = PPO(MlpPolicy, env, verbose=1, gamma=1, n_steps = epLen)
model.learn(total_timesteps=10)

env = gym.make('Ambulance-v1')
n_episodes = 2
res_mean, res_std = evaluate_policy(model, env, n_eval_episodes=n_episodes)

print(res_mean, '+/-', 1.96*res_std/np.sqrt(n_episodes))