'''
Implementation of a basic RL environment for continuous spaces.
Includes three test problems which were used in generating the figures.
'''

import numpy as np
import gym
from gym import spaces
import math

#-------------------------------------------------------------------------------


class Environment(object):
    '''General RL environment'''

    def __init__(self):
        pass

    def reset(self):
        pass

    def advance(self, action):
        '''
        Moves one step in the environment.

        Args:
            action

        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        return 0, 0, 0


#-------------------------------------------------------------------------------

'''Implementation of a continuous environment using the AI Gym from Google'''
class ContinuousAIGym(Environment):
    def __init__(self, env, epLen):
        '''
            env - AI Gym Environment
            epLen - Number of steps per episode
        '''
        self.env = env
        self.epLen = epLen
        self.timestep = 0
        self.state = self.env.reset()


    def get_epLen(self):
        return self.epLen

    def reset(self):
        '''Reset the environment'''
        self.timestep = 0
        self.state = self.env.reset()

    def advance(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action
        Returns:
            reward - double - reward
            newState - int - new state
            pContinue - 0/1 - flag for end of the episode
        '''
        newState, reward, terminal, info = self.env.step(action)

        if self.timestep == self.epLen or terminal:
            pContinue = 0
            self.reset()
        else:
            pContinue = 1

        return reward, newState, pContinue
