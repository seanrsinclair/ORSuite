import numpy as np

import sklearn_extra.cluster

import sys
from .. import Agent

''' Agent that implements a mode heuristic algorithm for the ambulance graph environment'''
class modeAgent(Agent):

    def __init__(self, epLen):
        '''
        epLen - number of steps
        data - all data observed so far
        call_locs - the node locations of all calls observed so far
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
        # After that, choose the locations where calls have occurred most frequently
        # in the past
        if timestep == 0:
            return state
        else:
            num_ambulance = len(self.data[0])
            counts = np.bincount(self.call_locs)
            action = []
            for i in range(num_ambulance):
                mode = np.argmax(counts)
                action.append(mode)
                counts[mode] = 0
            return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
