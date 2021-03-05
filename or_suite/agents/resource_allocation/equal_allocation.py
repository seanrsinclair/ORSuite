import numpy as np

from .. import Agent

''' Agent which implements several heuristic algorithms'''
class equalAllocationAgent(Agent):

    def __init__(self, epLen):
        '''args:
            epLen - number of steps
            func - function used to decide action
            data - all data observed so far
        '''
        self.epLen = epLen
        self.data = []

    def reset(self):
        # resets data matrix to be empty
        self.data = []

    def update_obs(self, obs, action, reward, newObs, timestep, info): 
        '''Add observation to records'''
        self.data.append(newObs)
        return

    def get_num_arms(self):
        return 0

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''
        
        budget,_,rel_endowment = state
        try:  
            print(self.data[0])
            action = rel_endowment*self.data[0][:len(budget)]
        except:
            action = rel_endowment*budget
        return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
