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


DEFAULT_CONFIG =  or_suite.envs.env_configs.ambulance_metric_default_config


agents = {'Random': or_suite.agents.rl.random.randomAgent(), 'Stable': or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen']), 'Median': or_suite.agents.ambulance.median.medianAgent(DEFAULT_CONFIG['epLen'])}
nEps = 50
numIters = 20
epLen = 5
DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance_graph/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : 5}

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

            CONFIG = DEFAULT_CONFIG
            CONFIG['alpha'] = alpha
            CONFIG['arrival_dist'] = arrival_dist
            DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_metric_'+str(agent)+'_'+str(alpha)+'_'+str(arrival_dist.__name__)+'/'
            ambulance_graph_env = gym.make('Ambulance-v0', config=CONFIG)

            run_single_algo(ambulance_graph_env, agents[agent], DEFAULT_SETTINGS)



######## Testing with Stable Baselines3 PPO Algorithm ########

env = make_vec_env('Ambulance-v1', n_envs=4)
model = PPO(MlpPolicy, env, verbose=1, gamma=1)
model.learn(total_timesteps=1000)

env = gym.make('Ambulance-v1')
n_episodes = 100
res_mean, res_std = evaluate_policy(model, env, n_eval_episodes=n_episodes)

print(-res_mean, '+/-', 1.96*res_std/np.sqrt(n_episodes))


