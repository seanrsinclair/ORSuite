import numpy as np
import sys
from .. import Agent

class commandLineAgent(Agent):

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
        num_ambulance = len(state)
        action = np.zeros(num_ambulance)
        for ambulance in range(num_ambulance):
          done = False
          while(not done):
            text = "Where do you want to position ambulance " + str(ambulance+1) + "? (choose a number between 0 and 1)"
            new_loc = input(text)
            try:
              new_loc = float(new_loc)
              if new_loc < 0 or new_loc > 1:
                raise ValueError
              action[ambulance] = new_loc
              done = True
            except ValueError:
              print("Please enter a number between 0 and 1")
        
        return action

    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
