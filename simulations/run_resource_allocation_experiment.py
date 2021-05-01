import sys

sys.path.append('../')

import numpy as np
import copy
import gym

import or_suite

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy


''' Defining parameters to be used in the experiment'''


# #TODO: Edit algo-list to be the names of the algorithms you created
problem_config_list = {'simple': or_suite.envs.env_configs.resource_allocation_simple_config, 
                'simple_poisson': or_suite.envs.env_configs.resource_allocation_simple_poisson_config,
                'multi': or_suite.envs.env_configs.resource_allocation_default_config
                }




for problem in problem_config_list:
    nEps = 50
    numIters = 50
    #initialize resource allocation environment w/ default parameters
    
    env = gym.make('Resource-v0', config = problem_config_list[problem])
    epLen = env.epLen
    # algo_information = {'Random': or_suite.agents.rl.random.randomAgent(), 'Equal_Allocation': or_suite.agents.resource_allocation.equal_allocation.equalAllocationAgent(epLen, DEFAULT_ENV_CONFIG)}
    algo_information = { 'HopeGuardrail': or_suite.agents.resource_allocation.hope_guardrail.hopeguardrailAgent(epLen, problem_config_list[problem], 1/2),
                        'EqualAllocation': or_suite.agents.resource_allocation.equal_allocation.equalAllocationAgent(epLen, problem_config_list[problem]),
                        'FixedThreshold': or_suite.agents.resource_allocation.fixed_threshold.fixedThresholdAgent(epLen, problem_config_list[problem])
                        }

    DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'render': False, 'dirPath': '../data/allocation/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': True, 'epLen' : epLen}


    path = {}
    path_list = []
    algo_list = []

    for agent in algo_information:
        print('#### NEW EXPERIMENT ####')
        print(agent)
        print(problem)
        print('####')
        algorithm = algo_information[agent]
        path_list.append('../data/allocation_%s_%s'%(agent,problem))
        algo_list.append(str(agent))
        DEFAULT_SETTINGS['dirPath'] = '../data/allocation_%s_%s'%(agent,problem)
        or_suite.utils.run_single_algo(env, algorithm, DEFAULT_SETTINGS)

    fig_path = '../figures/'
    fig_name = 'allocation_{}_line_plot.pdf'.format(problem)
    or_suite.plots.plot_line_plots(path_list, algo_list, fig_path, fig_name, int(nEps / 40)+1)

    fig_radar_name = 'allocation_{}_radar_plot.pdf'.format(problem)

    additional_metric = {'Waste': lambda traj : or_suite.utils.delta_EFFICIENCY(traj, problem_config_list[problem]),
                        'Envy': lambda traj : or_suite.utils.delta_HINDSIGHT_ENVY(traj, problem_config_list[problem]),
                        'Prop': lambda traj : or_suite.utils.delta_PROP(traj, problem_config_list[problem]),
                        'OPT': lambda traj : or_suite.utils.delta_COUNTERFACTUAL_ENVY(traj, problem_config_list[problem])}

    or_suite.plots.plot_radar_plots(path_list, algo_list, fig_path, fig_radar_name, additional_metric)


# #below is work on the PPO algorithm, kinda not the greatest atm
# env = gym.make('Resource-v0', config = DEFAULT_ENV_CONFIG)
# model = PPO(MlpPolicy, env, verbose=1, gamma=1)
# model.learn(total_timesteps=1000)

# env = gym.make('Ambulance-v1')
# n_episodes = 100
# res_mean, res_std = evaluate_policy(model, env, n_eval_episodes=n_episodes)

# print(-res_mean, '+/-', 1.96*res_std/np.sqrt(n_episodes))