import numpy as np
import sys


import sys
from .. import Agent

''' Agent that implements a k-medoid heuristic algorithm for the line ambulance environment'''
class medianAgent(Agent):

    def __init__(self, epLen):
        '''
        epLen - number of steps
        data - all data observed so far
        call_locs - the locations of all calls observed so far
        '''
        self.epLen = epLen
        self.data = []
        self.call_locs = []

    def reset(self):
        # Resets data and call_locs arrays to be empty
        self.data = []
        self.call_locs = []

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

        # Adds the most recent state obesrved in the environment to data
        self.data.append(newObs)

        # Adds the most recent arrival location observed to call_locs
        self.call_locs.append(info['arrival'])
        return

    def update_policy(self, k):
        '''Update internal policy based upon records'''

        # Greedy algorithm does not update policy
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''

        # For the first iteration, choose the starting state
        # After that, choose locations for the ambulances that are most centrally
        # located to the locations of previous calls using the k-medoids algorithm
        # For more details about the k-medoids algorithm, see the readme document
        # for the ambulance environment or the sci-kit learn documentation
        if len(self.data) == 0:
            return state
        else:
            num_ambulance = len(self.data[0])
            # print(num_ambulance)
            left_points = [(1 / num_ambulance)*i for i in range(num_ambulance)]
            quantiles = []
            for j in range(len(left_points)):
                if j == len(left_points) - 1:
                    quantiles.append(((1 - left_points[j]) / 2) + left_points[j])
                else:
                    quantiles.append(((left_points[j+1] - left_points[j]) / 2) + left_points[j])
            # print(quantiles)
            action = np.quantile(np.asarray(self.call_locs), quantiles)
            # print(action)
            return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
