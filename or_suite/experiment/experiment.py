import time
from shutil import copyfile
import pandas as pd
import tracemalloc
import numpy as np
import os

class Experiment(object):

    def __init__(self, env, agent, dict):
        '''
        A simple class to run a MDP Experiment.
        Args:
            env - an instance of an Environment
            agent - an agent
            dict - a dictionary containing the arguments to send for the experiment, including:
                seed - random seed for experiment
                recFreq - proportion of episodes to save to file
                targetPath - path to the file for saving
                deBug - boolean of whether to include
                nEps - number of episodes
                numIters - the number of iterations to run experiment
                saveTrajectory - boolean of whether to save trajectory information
        '''
        # assert isinstance(env, environment.Environment)

        self.seed = dict['seed']
        self.epFreq = dict['recFreq']
        self.targetPath = dict['targetPath']
        self.deBug = dict['deBug']
        self.nEps = dict['nEps']
        self.env = env
        self.epLen = dict['epLen']
        self.num_iters = dict['numIters']
        self.save_trajectory = dict['saveTrajectory']
        self.agent = agent
        # print('epLen: ' + str(self.epLen))
        self.data = np.zeros([dict['nEps']*self.num_iters, 5])


        if self.save_trajectory:
            self.trajectory = np.zeros([dict['nEps']*self.num_iters*self.epLen, 8])

        np.random.seed(self.seed)

    # Runs the experiment
    def run(self):
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')


        index = 0
        traj_index = 0
        for i in range(self.num_iters):
            self.agent.reset()
            self.agent.update_config(self.env.get_config())
            for ep in range(1, self.nEps+1):
                # print('Episode : ' + str(ep))
                # Reset the environment
                self.env.reset()
                oldState = self.env.state
                epReward = 0

                self.agent.update_policy(ep)

                pContinue = True
                h = 0

                start_time = time.time()
                tracemalloc.start()

                while pContinue and h < self.epLen:
                    # Step through the episode
                    if self.deBug:
                        print('state : ' + str(oldState))
                    action = self.agent.pick_action(oldState, h)
                    if self.deBug:
                        print('action : ' + str(action))

                    newState, reward, pContinue, info = self.env.step(action)
                    epReward += reward

                    self.agent.update_obs(oldState, action, reward, newState, h, info)

                    if self.save_trajectory: # TODO: state, action, reward, etc are not necessarily numbers - so what is the best way of storing this in a list?
                        self.trajectory[traj_index, 0] = i
                        self.trajectory[traj_index, 1] = ep
                        self.trajectory[traj_index, 2] = h
                        self.trajectory[traj_index, 3] = oldState
                        self.trajectory[traj_index, 4] = action
                        self.trajectory[traj_index, 5] = reward
                        self.trajectory[traj_index, 6] = newState
                        self.trajectory[traj_index, 7] = info
                    traj_index += 1

                    oldState = newState
                    h = h + 1

                current, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                end_time = time.time()
                
                if self.deBug:
                    print('final state: ' + str(newState))
                # print('Total Reward: ' + str(epReward))

                # Logging to dataframe
                # if ep % self.epFreq == 0:
                # print('## LOGGING TO DATA FRAME ##')
                # print('Episode : ' + str(ep))
                # print('Total Reward: ' + str(epReward))
                # print('##                       ##')
                self.data[index, 0] = ep-1
                self.data[index, 1] = i
                self.data[index, 2] = epReward
                self.data[index, 3] = current
                self.data[index, 4] = ((end_time) - (start_time))

                index += 1

        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')

    # Saves the data to the file location provided to the algorithm
    def save_data(self , dir_path, targetPath): # TODO: Best way of getting directory locations for both paths?
        print('**************************************************')
        print('Saving data')
        print('**************************************************')

        print(self.data)


        if self.save_trajectory:
            data_loc, traj_loc = targetPath

            dt = pd.DataFrame(self.data, columns=['episode', 'iteration', 'epReward', 'memory', 'time'])
            dt = dt[(dt.T != 0).any()]

            traj = pd.DataFrame(self.trajectory, columns=['index', 'episode', 'step', 'oldState', 'reward', 'newState', 'info'])
            print('Writing to file ' + data_loc)
        else:
            data_loc = targetPath

            dt = pd.DataFrame(self.data, columns=['episode', 'iteration', 'epReward', 'memory', 'time'])
            dt = dt[(dt.T != 0).any()]
            print('Writing to file ' + data_loc)

        if os.path.exists(dir_path):
            dt.to_csv(os.path.join(dir_path,data_loc), index=False, float_format='%.2f', mode='a')
            if self.save_trajectory:
                dt.to_csv(os.path.join(dir_path, traj_loc), index=False, float_format='%.2f', mode='a')
        else:
            os.makedirs(dir_path)
            dt.to_csv(os.path.join(dir_path, data_loc), index=False, float_format='%.2f')
            if self.save_trajectory:
                dt.to_csv(os.path.join(dir_path, traj_loc), index=False, float_format='%.2f', mode='a')

        print('**************************************************')
        print('Data save complete')
        print('**************************************************')

        return dt