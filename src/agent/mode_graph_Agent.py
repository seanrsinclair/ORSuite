import numpy as np
import agent
import sklearn_extra.cluster


''' Agent which implements several heuristic algorithms'''
class modeAgent(agent.FiniteHorizonAgent):

    def __init__(self, epLen):
        '''args:
            epLen - number of steps
            func - function used to decide action
            data - all data observed so far
            alpha - alpha parameter in ambulance problem
        '''
        self.epLen = epLen
        self.data = []
        self.call_locs = []

    def reset(self):
        # resets data matrix to be empty
        self.data = []
        self.call_locs = []

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        self.data.append(newObs)
        self.call_locs.append(info['arrival'])
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

        #choose locations where calls occur most frequently

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
