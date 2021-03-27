import sys

sys.path.append('../')

import numpy as np
import gym

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
    _ = exp.run()
    dt_data = exp.save_data()


DEFAULT_CONFIG = or_suite.envs.env_configs.ambulance_graph_default_config

agents = {'Random': or_suite.agents.rl.random.randomAgent(), 'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']), 'Median': or_suite.agents.ambulance.median_graph.medianAgent(DEFAULT_CONFIG['epLen'], DEFAULT_CONFIG['edges'], DEFAULT_CONFIG['num_ambulance']), 'Mode': or_suite.agents.ambulance.mode_graph.modeAgent(DEFAULT_CONFIG['epLen']), 'SB_PPO': None}
# agents = {'Random': or_suite.agents.rl.random.randomAgent(), 'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']), 'Median': or_suite.agents.ambulance.median_graph.medianAgent(DEFAULT_CONFIG['epLen'], DEFAULT_CONFIG['edges'], DEFAULT_CONFIG['num_ambulance']), 'Mode': or_suite.agents.ambulance.mode_graph.modeAgent(DEFAULT_CONFIG['epLen'])}

# agents = {'SB_PPO': None}
nEps = 1000
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

arrival_dists = [uniform, nonuniform]


for agent in agents:
    for num_ambulance in num_ambulances:
        for alpha in alphas:
            for arrival_dist in arrival_dists:
                print(agent)
                print(num_ambulance)
                print(alpha)
                print(arrival_dist.__name__)
                CONFIG = DEFAULT_CONFIG
                CONFIG['alpha'] = alpha
                CONFIG['arrival_dist'] = arrival_dist
                CONFIG['num_ambulance'] = num_ambulance
                CONFIG['starting_state'] = [0 for _ in range(num_ambulance)]
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
                    run_single_algo(ambulance_graph_env, agents[agent], DEFAULT_SETTINGS)
for num_ambulance in num_ambulances:
    for alpha in alphas:
        for arrival_dist in arrival_dists:
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


######## Testing with Stable Baselines3 PPO Algorithm ########


