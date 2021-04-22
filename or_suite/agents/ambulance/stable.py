import numpy as np

import sys
from .. import Agent


class stableAgent(Agent):
    """
    Agent that implements a stable heuristic algorithm for use with either ambulance environment
    
    Methods:
        reset() : clears data and call_locs which contain data on what has occurred so far in the environment
        update_config() : (UNIMPLEMENTED)
        update_obs(obs, action, reward, newObs, timestep, info) : 
            adds newObs, the most recently observed state, to data
            adds the most recent call arrival, found in info['arrival'] to call_locs
        update_policy() : not used, because a greedy algorithm does not have a policy
        pick_action(state, step) : the action chosen is the most recently obesrved state
            of the environment

    Attributes:
        epLen: (int) number of time steps to run the experiment for
        data: (int list list or float list list) a list of all the states of the environment observed so far
    
    """

    def __init__(self, epLen):
        '''
        epLen - number of time steps
        data - all data observed so far
        '''
        self.epLen = epLen
        self.data = []

    def reset(self):
        # Resets data array to be empty
        self.data = []

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

        # Adds the most recent state obesrved in the environment to data
        self.data.append(newObs)
        return

    def update_policy(self, k):
        '''Update internal policy based upon records'''

        # Greedy algorithm does not update policy
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Chooses the most recently observed state as the action.
        '''

        # For the first iteration, choose the starting state
        # After that, choose the most recently observed state as the new location 
        # for each ambulance. This results in no ambulance movement between calls
        if len(self.data) == 0:
            return state
        else:
            action = self.data[-1]
            return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
