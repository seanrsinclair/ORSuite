import sys

sys.path.append('../')

import numpy as np
import gym

import or_suite

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy



DEFAULT_CONFIG =  or_suite.envs.env_configs.ambulance_metric_default_config
DEFAULT_CONFIG['alpha']=0.25

nEps = 50
numIters = 20
epLen = 5


######## Testing with Stable Baselines3 PPO Algorithm ########

env = gym.make('Ambulance-v0', config = DEFAULT_CONFIG)
model = PPO(MlpPolicy, env, verbose=1, gamma=(epLen - 1) / epLen)
model.learn(total_timesteps=5000,  eval_freq = 1, eval_log_path = '../data/')
# model.save_replay_buffer("sac_replay_buffer")


n_episodes = 100
res_mean, res_std = evaluate_policy(model, env, n_eval_episodes=n_episodes)

print(-res_mean, '+/-', 1.96*res_std/np.sqrt(n_episodes))


