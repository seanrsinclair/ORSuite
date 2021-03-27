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



def run_single_algo(env, agent, settings):

    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()


DEFAULT_CONFIG =  or_suite.envs.env_configs.ambulance_metric_default_config
epLen = DEFAULT_CONFIG['epLen']
nEps = 100
numIters = 1

epsilon = (nEps * epLen)**(-1 / 4)
action_net = np.arange(start=0, stop=1, step=epsilon)
state_net = np.arange(start=0, stop=1, step=epsilon)

scaling = 0.5

agents = {# 'SB_PPO': None, 'Random': or_suite.agents.rl.random.randomAgent(),
     # 'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']),
     # 'Median': or_suite.agents.ambulance.median.medianAgent(DEFAULT_CONFIG['epLen']), 
     # 'Unif_MB': or_suite.agents.rl.eNet_model_Agent.eNetModelBased(action_net, state_net, epLen, scaling, 0, False),
     # 'Unif_QL': or_suite.agents.rl.eNet_Agent.eNet(action_net, state_net, epLen, scaling),
     # 'AdaQL': or_suite.agents.rl.adaptive_Agent.AdaptiveDiscretization(epLen, numIters, scaling),
     # 'AdaMB': or_suite.agents.rl.adaptive_model_Agent.AdaptiveModelBasedDiscretization(epLen, numIters, scaling, 0, 2, True, True)
     # 'Unif_QL': or_suite.agents.rl.eNet_Multiple.eNet(action_net, state_net, epLen, scaling, (3,3))
     'Unif_MB': or_suite.agents.rl.eNet_model_Agent_Multiple.eNetModelBased(action_net, state_net, epLen, scaling, (3,3), 0, True)
     }



DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : 5}

alphas = [0, 0.25, 1]

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
# arrival_dists = [beta]
# num_ambulances = [1,3]
num_ambulances = [3]

for agent in agents:
    for num_ambulance in num_ambulances:
        for alpha in alphas:
            for arrival_dist in arrival_dists:
                print(agent)
                print(alpha)
                print(arrival_dist.__name__)
                CONFIG = copy.deepcopy(DEFAULT_CONFIG)
                CONFIG['alpha'] = alpha
                CONFIG['arrival_dist'] = arrival_dist
                CONFIG['num_ambulance'] = num_ambulance
                CONFIG['starting_state'] = np.array([0 for _ in range(num_ambulance)])
                DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/'
                ambulance_env = gym.make('Ambulance-v0', config=CONFIG)

                if agent == 'SB_PPO':
                    episodes = []
                    iterations = []
                    rewards = []
                    times = []
                    memory = []
                
                    for i in range(numIters):
                        sb_env = Monitor(ambulance_env)
                        model = PPO(MlpPolicy, sb_env, gamma=1, verbose=0, n_steps = epLen)
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
                    run_single_algo(ambulance_env, agents[agent], DEFAULT_SETTINGS)




for num_ambulance in num_ambulances:
    for alpha in alphas:
        for arrival_dist in arrival_dists:
            path_list_line = []
            algo_list_line = []

            path_list_radar = []
            algo_list_radar = []
            for agent in agents:
                path_list_line.append('../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/data.csv')
                algo_list_line.append(str(agent))
                if agent != 'SB_PPO':
                    path_list_radar.append('../data/ambulance_metric_'+str(agent)+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/data.csv')
                    algo_list_radar.append(str(agent))

            fig_path = '../figures/'
            fig_name = 'ambulance_metric'+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'_line_plot'+'.pdf'
            or_suite.plots.plot_line_plots(path_list_line, algo_list_line, fig_path, fig_name)


            fig_name = 'ambulance_metric'+'_'+str(num_ambulance)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'_radar_plot'+'.pdf'
            or_suite.plots.plot_radar_plots(path_list_radar, algo_list_radar, fig_path, fig_name)

# ######## Testing with Stable Baselines3 PPO Algorithm ########

# env = make_vec_env('Ambulance-v1', n_envs=4)
# model = PPO(MlpPolicy, env, verbose=1, gamma=1)
# model.learn(total_timesteps=1000)

# env = gym.make('Ambulance-v1')
# n_episodes = 100
# res_mean, res_std = evaluate_policy(model, env, n_eval_episodes=n_episodes)

# print(-res_mean, '+/-', 1.96*res_std/np.sqrt(n_episodes))


