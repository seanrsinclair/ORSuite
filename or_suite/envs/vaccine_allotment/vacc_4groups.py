# Implementation of a basic RL environment for discrete spaces. 

import numpy as np
import gym
from gym import spaces
import math
import random
from .. import dynamics_model_4groups as dm4g
from .. import env_configs

#------------------------------------------------------------------------------
'''
A simple vaccine distribution environment.  
Agent interacts w/ environment by picking a vaccine priority group order for vaccine distribution to a population over a set time period. 
Vaccines are distributed to the first priority group until there are no vaccines left or no people in the first priority group. 
If there are still vaccines available, they are distributed to the next priority group. 
We go down the priority list until vaccine supply is exhausted or there are no vaccine candidates left. 
There is a penalty for new infections in this time period.
Only 4 groups are considered (e.g. medical workers, essential non-medical workers, low-risk, others)
    - TOTAL population size and vaccine supply for each time period is kept constant. 
    - Each group is split into susceptible (S), asymptomatically infected (A)
    - We keep track of agreggate mildly symptomatically infected (I) and hospitalized (H) individuals. 
'''

#------------------------------------------------------------------------------
class VaccineEnvironment(gym.Env):
    """
    A simple vaccine distribution environment with a discrete action and observation space.
    
    Methods:
        reset() : reinitializes timestep and state returns the starting state
        get_config() : returns the current configuration
        step(action) : implements a step in the RL environment
        render(mode) : (UNIMPLEMENTED)
        close() : (UNIMPLEMENTED)
        
    Attributes: TODO
    
    """
    # don't worry about this, has to do with how gym exports text/info to the termial
    metadata = {'render.modes': ['human']}


    def __init__( self, config = env_configs.vaccine_4groups_default_config):
        """
        Creates a VaccineEnvironment object.
        
        Arguments: 
            config: dictionary with the following keys (and corresponding values)
                - epLen : (int) number of time steps 
                - starting_state : (np.array) initial population group sizes; should contain 11 entries
                - parameters : (dict) of parameter values to pass to dynamics model
        For more detailed information, see the file vaccine_allocation_readme.ipynb
        
        Typical usage example:
        TODO
        """
        
        self.config = config
        self.epLen = config['epLen']
        self.vaccines = config['parameters']['vaccines']
        self.priority_order = config['parameters']['priority_order']
        self.parameters = config['parameters']
        self.total_pop = np.sum(config['starting_state'])
        self.state = config['starting_state']
        self.starting_state = config['starting_state']
        self.timestep = 0
            
        '''
        Action Space (finite):
        - set of all possible actions (priority orders in this case)
        - An action is an index i and the corresponding permutation is all_priority_orders[i]
        - 0 corresponds to [], which means no priority order
        See vaccine_allocation_readme.ipynb for more detail.
        '''
        self.action_space = spaces.Discrete(25)
        self.all_priority_orders = [[], ["c1","c2","c3","c4"],["c2","c1","c3","c4"],["c3","c1","c2","c4"],["c1","c3","c2","c4"],
                               ["c2","c3","c1","c4"],["c3","c2","c1","c4"],["c4","c2","c1","c3"],["c2","c4","c1","c3"],
                               ["c1","c4","c2","c3"],["c4","c1","c2","c3"],["c2","c1","c4","c3"],["c1","c2","c4","c3"],
                               ["c1","c3","c4","c2"],["c3","c1","c4","c2"],["c4","c1","c3","c2"],["c1","c4","c3","c2"],
                               ["c3","c4","c1","c2"],["c4","c3","c1","c2"],["c4","c3","c2","c1"],["c3","c4","c2","c1"],
                               ["c2","c4","c3","c1"],["c4","c2","c3","c1"],["c3","c2","c4","c1"],["c2","c3","c4","c1"]] 
    
        '''
        Observation space (finite):
        A tuple of integer values representing certain population stats. 
        See vaccine_allocation_readme.ipynb for more detail.
        '''
                                 
        
        # The obersvation/state space is a spaces.MultiDiscrete object
        self.observation_space = spaces.MultiDiscrete( ([self.total_pop+1]*11) )
    
    
    def reset(self):
        """
        Reinitializes variables and returns the starting state.
        """
        self.timestep = 0
        self.state = self.starting_state
        return self.starting_state
      
      
    def get_config(self):
        """
        Returns the current configuration.
        """
        return self.config


    def step(self, action):
        """
        Moves one step in the environment.

        Arguments:
            action - int - chosen action 
        
        Returns:
            reward - double - reward based on chosen action
            newState - np.array of integers - new state
            done - 0/1 - flag for end of the episode
            info - dict - information we can use to plot things related to disease dynamics
        """
        assert self.action_space.contains(action),"Action is invalid"
        
        old_state = self.state
        # print('old_state' , old_state)
        
        self.priority_order = self.all_priority_orders[action]
        self.parameters['priority_order'] = self.priority_order
        
        newState, info = dm4g.dynamics_model(self.parameters, self.state)
        # print('New state' , newState)
        
        # 'reward' is number of new infections times -1
        reward = float(-1*newState[len(newState)-1])

        if self.timestep != (self.epLen-1):
            done = False
        else:
            done = True

        self.state = newState
        self.timestep += 1
        
        return self.state, reward,  done, info


    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()


    def close(self):
        pass


