import sys

sys.path.append('../')


import numpy as np
import gym

import or_suite

def run_single_algo(env, agent, settings): 

    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()




DEFAULT_CONFIG = {'epLen': 5, 'arrival_dist': None, 'alpha': 0.25,
                'edges': [(0,4,{'dist':7}), (0,1,{'dist':1}), (1,2,{'dist':3}), (2,3,{'dist':5}), (1,3,{'dist':1}), (1,4,{'dist':17}), (3,4,{'dist':3})],
                'starting_state': [1,2], 'num_ambulance': 2}



# agents = [or_suite.agents.ambulance.stableAgent(DEFAULT_CONFIG['epLen']), or_suite.agents.ambulance.medianAgent(DEFAULT_CONFIG['epLen'], DEFAULT_CONFIG['edges'], DEFAULT_CONFIG['num_ambulance']), modeAgent(DEFAULT_CONFIG['epLen'])]
agents = [or_suite.agents.ambulance.stable.stableAgent(DEFAULT_CONFIG['epLen'])]

nEps = 500
numIters = 50
DEFAULT_SETTINGS = {'seed': 1, 'recFreq': 1, 'dirPath': '../data/ambulance_graph/', 'deBug': False, 'nEps': nEps, 'numIters': numIters, 'saveTrajectory': False, 'epLen' : 5}

alphas = [0, 1, 0.25]
arrival_dists = [None, [0.25, 0.4, 0.25, 0.05, 0.05]]


# for agent in agents:
#     for alpha in alphas:
#         for arrival_dist in arrival_dists:
#             agent.reset()

#             CONFIG = DEFAULT_CONFIG
#             CONFIG['alpha'] = alpha
#             CONFIG['arrival_dist'] = arrival_dist
#             DEFAULT_SETTINGS['dirPath'] = '../data/ambulance_graph_'+str(alpha)+'_'+str(arrival_dist)+'/'
#             ambulance_graph_env = gym.make('Ambulance-v1', config=CONFIG)

#             run_single_algo(ambulance_graph_env, agent, DEFAULT_SETTINGS)


from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy



def evaluate(model, num_episodes=100):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :return: (float) Mean reward for the last num_episodes
    """
    # This function will only work for a single Environment
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            # _states are only useful when using LSTM policies
            action, _states = model.predict(obs)
            # here, action, rewards and dones are arrays
            # because we are using vectorized env
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward, np.std(all_episode_rewards)



CONFIG = DEFAULT_CONFIG
CONFIG['alpha'] = alphas[0]
CONFIG['arrival_dist'] = arrival_dists[0]


env = gym.make('Ambulance-v1', config=CONFIG)
model = PPO(MlpPolicy, env, verbose=0, n_steps = DEFAULT_CONFIG['epLen'], gamma = 1)


# Random Agent, before training
mean_reward, std_reward = evaluate(model)

print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")


model.learn(total_timesteps=1000)

mean_reward, std_reward = evaluate(model)
print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")







