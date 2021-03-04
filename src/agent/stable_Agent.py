import numpy as np
import agent


''' Agent that implements a stable heuristic algorithm for use with either ambulance environment'''
class stableAgent(agent.FiniteHorizonAgent):

    def __init__(self, epLen):
        '''
        TODO: epLen - number of time steps
        TODO: func - function used to decide action
        data - all data observed so far
        TODO: alpha - alpha parameter in ambulance problem
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

    def get_num_arms(self):
        return 0

    def update_policy(self, k):
        '''Update internal policy based upon records'''

        # Greedy algorithm does not update policy
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''

        # For the first iteration, choose the starting state
        # After that, choose the most recently observed state as the new location 
        # for each ambulance. This results in no ambulance movement between calls
        if timestep == 0:
            return state
        else:
            action = self.data[-1]
            return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
