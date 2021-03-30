import time
from shutil import copyfile
import pandas as pd
import tracemalloc
import numpy as np
import pickle
import os
from stable_baselines3.common.monitor import Monitor


class SB_Experiment(object):

    def __init__(self, env, model, dict):
        '''
        A simple class to run a MDP Experiment with a stable baselines model.
        Args:
            env - an instance of an Environment
            model - a stable baselines model
            dict - a dictionary containing the arguments to send for the experiment, including:
                seed - random seed for experiment
                recFreq - proportion of episodes to save to file
                targetPath - path to the file for saving
                deBug - boolean of whether to include
                nEps - number of episodes
                numIters - the number of iterations to run experiment
                saveTrajectory - boolean of whether to save trajectory information
        '''


        self.seed = dict['seed']
        self.epFreq = dict['recFreq']
        self.dirPath = dict['dirPath']
        # self.targetPath = dict['targetPath']
        self.deBug = dict['deBug']
        self.nEps = dict['nEps']
        self.env = env
        self.epLen = dict['epLen']
        self.num_iters = dict['numIters']
        self.save_trajectory = dict['saveTrajectory']
        self.model = model
        # print('epLen: ' + str(self.epLen))


        if self.save_trajectory:
            self.trajectory = []

        np.random.seed(self.seed)

    # Runs the experiment
    def run(self):
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')


        index = 0
        traj_index = 0
        episodes = []
        iterations = []
        rewards = []
        times = []
        memory = []
        
        # Running an experiment

        # TODO: Determine how to save trajectory information
        for i in range(self.num_iters):
            tracemalloc.start()

            self.model.learn(total_timesteps=self.epLen*self.nEps)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            episodes = np.append(episodes,np.arange(0, self.nEps))
            iterations = np.append(iterations, [i for _ in range(self.nEps)])

            memory = np.append(memory, [current for _ in range(self.nEps)])

        rewards = np.append(rewards, self.env.get_episode_rewards())

        # Times are calculated cumulatively so need to calculate the per iteration time complexity
        orig_times = [0.] + self.env.get_episode_times()
        times = [orig_times[i] - orig_times[i-1] for i in np.arange(1, len(orig_times))]

        # Combining data in dataframe
        self.data = pd.DataFrame({'episode': episodes,
                            'iteration': iterations,
                            'epReward': rewards,
                            'time': np.log(times),
                            'memory': memory})
        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')

    # Saves the data to the file location provided to the algorithm
    def save_data(self):

        print('**************************************************')
        print('Saving data')
        print('**************************************************')

        print(self.data)

        dir_path = self.dirPath

        data_loc = 'data.csv'


    
        dt = self.data
        dt = dt[(dt.T != 0).any()]

        print('Writing to file ' + dir_path + data_loc)
    
        
        if os.path.exists(dir_path):
            dt.to_csv(os.path.join(dir_path,data_loc), index=False, float_format='%.5f', mode='w')

        else:
            os.makedirs(dir_path)
            dt.to_csv(os.path.join(dir_path, data_loc), index=False, float_format='%.5f', mode='w')


        print('**************************************************')
        print('Data save complete')
        print('**************************************************')

        return dt