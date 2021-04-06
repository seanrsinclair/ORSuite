# Implementation of a basic RL environment for discrete spaces. 

import numpy as np
import gym
from gym import spaces
import math
import random
import dynamics_model_4groups as dm4g
from .. import env_configs

#------------------------------------------------------------------------------
'''A simple vaccine distribution environment.  
Agent interacts w/ environment by picking a vaccine priority group order for vaccine distribution to a population over a set time period. 
Vaccines are distributed to the first priority group until there are no vaccines left or no people in the first priority group. 
If there are still vaccines available, they are distributed to the next priority group. 
We go down the priority list until vaccine supply is exhausted or there are no vaccine candidates left. 
There is a penalty for new infections in this time period.
Only 4 groups are considered (e.g. medical workers, essential non-medical workers, low-risk, others)'''

# TODO: these configs should be from env_configs
MODEL_PARAMETERS = {'contact_matrix': np.tril(np.ones((4,4)))*0.2, 'lambda_hosp': 0.4,'rec': 0, 'p1': 0.3, 'p2': 0.3, 'p3': 0.6, 'p4': 0.3,
                        'h1': 0.1, 'h2': 0.1, 'h3': 0.5,'h4': 0.1, 'gamma': 50, 'beta': 1.5, 'priority_order': [], 'vaccines': 400, 'time_step': 14}

DEFAULT_CONFIG = {'epLen': 6, 'alpha': 0.25, 'starting_state': np.array([1990, 1990, 1990, 1990, 10, 10, 10, 10, 0, 0]), 'parameters': MODEL_PARAMETERS}

#------------------------------------------------------------------------------
class VaccineEnvironment(gym.Env):
    """
    Custom Environment that follows gym interface.
    - TOTAL population size and vaccine supply for each time period is kept constant. 
    - There are four possible groups (risk classes) total.
    -- Each group is split into susceptible (S), asymptomatically infected (A)
    -- We keep track of agreggate mildly symptomatically infected (I) and hospitalized (H) individuals. 
    """
    # don't worry about this, has to do with how gym exports text/info to the termial
    metadata = {'render.modes': ['human']}


    def __init__( self, config = DEFAULT_CONFIG):
    
        # TODO: shorten this; add more detail to readme file
        '''
        Input: a dictionary with the following keys (and corresponding values)
        o epLen - number of time steps (note that one single time step could be 1 day, 7 days, even a whole month)
        o alpha - parameter for difference in costs/loss
        o priority_order - starting priority order
        o starting_state - np.array of initial population group sizes
            ~ [s1, s2, s3, s4, a1, a2, a3, a4, I, H] where
                o s1-s4 = susceptible persons in groups 1-4
                o a1-a4 = asymptomatic persons in groups 1-4
                o I = total number of mildy symptomatic persons across all groups
                o H = total number of hospitalized persons across all groups
        o parameters - dictionary of parameter values to pass to dynamics model
        '''
        
        self.config = config
        self.epLen = config['epLen']
        self.vaccines = config['parameters']['vaccines']
        self.alpha = config['alpha']
        self.priority_order = config['parameters']['priority_order']
        self.parameters = config['parameters']
        self.total_pop = np.sum(config['starting_state'])
        self.state = config['starting_state']
        self.starting_state = config['starting_state']
        # self.new_infs = 0
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
        '''
        Reinitializes variables and returns the starting state
        '''
        self.timestep = 0
        self.state = self.starting_state
        return self.starting_state
      
      
    def get_config(self)
        return self.config


    def step(self, action):
        '''
        Move one step in the environment

        Args:
        action - int - chosen action 
        Returns:
            reward - double - reward based on chosen action
            newState - np.array of integers - new state
            done - 0/1 - flag for end of the episode
            info - dict - information we can use to plot things related to disease dynamics
        '''
        old_state = self.state
        # print('old_state' , old_state)
        
        self.priority_order = self.all_priority_orders[action]
        self.parameters['priority_order'] = self.priority_order
        
        newState, info = dm4g.dynamics_model(self.parameters, self.state)
        # print('New state' , newState)
        
        # 'reward' is number of new infections times -1
        reward = -1*newState[len(newState)-1]

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


