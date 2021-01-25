'''
Script to run simple continuous RL experiments.
'''
import sys


sys.path.append('../agent/')
sys.path.append('../environment/')

import numpy as np
import pandas as pd
import agent
import environment
import matplotlib.pyplot as plt
import os.path as path
from shutil import copyfile

class Experiment(object):

    def __init__(self, env, agent_list, dict):
        '''
        A simple class to run a MDP Experiment.

        Args:
            env - an instance of an Environment
            agent_list - a list of a Agents
            dict - a dictionary containing the arguments to send for the experiment, including:
                seed - random seed for experiment
                recFreq - proportion of episodes to save to file
                targetPath - path to the file for saving
                deBug - boolean of whether to include
                nEps - number of episodes
                numIters - the number of iterations to run experiment, should match
                number of agents in agent_list
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
        self.agent_list = agent_list
        # print('epLen: ' + str(self.epLen))
        self.data = np.zeros([dict['nEps']*self.num_iters, 3])

        np.random.seed(self.seed)

    # Runs the experiment
    def run(self):
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')


        index = 0
        for i in range(self.num_iters):
            agent = self.agent_list[i]
            for ep in range(1, self.nEps+1):
                # print('Episode : ' + str(ep))
                # Reset the environment
                self.env.reset()
                oldState = self.env.state
                epReward = 0

                agent.update_policy(ep)

                pContinue = True
                h = 0
                while pContinue and h < self.epLen:
                    # Step through the episode
                    if self.deBug:
                        print('state : ' + str(oldState))
                    action = agent.pick_action(oldState, h)
                    if self.deBug:
                        print('action : ' + str(action))

                    newState, reward, pContinue, info = self.env.step(action)
                    epReward += reward

                    agent.update_obs(oldState, action, reward, newState, h)
                    oldState = newState
                    h = h + 1
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
                index += 1

        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')

    # Saves the data to the file location provided to the algorithm
    def save_data(self):
        print('**************************************************')
        print('Saving data')
        print('**************************************************')

        print(self.data)

        dt = pd.DataFrame(self.data, columns=['episode', 'iteration', 'epReward'])
        dt = dt[(dt.T != 0).any()]
        print('Writing to file ' + self.targetPath)
        if path.exists(self.targetPath):
            dt.to_csv(self.targetPath, index=False, float_format='%.2f', mode='a')
        else:
            dt.to_csv(self.targetPath, index=False, float_format='%.2f')


        print('**************************************************')
        print('Data save complete')
        print('**************************************************')

        return dt
