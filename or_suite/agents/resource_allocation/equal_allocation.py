import numpy as np
from .. import Agent


''' Agent which implements several heuristic algorithms'''
class equalAllocationAgent(Agent):

    def __init__(self, epLen, env_config):
        '''args:
            epLen - number of steps
            func - function used to decide action
            env_config - parameters used in initialization of environment
            data - all data observed so far
        '''

        self.num_types = env_config['weight_matrix'].shape[0]
        self.num_resources = self.env_config['weight_matrix'].shape[1]
        
        self.epLen = epLen
        self.env_config = env_config
        self.data = []
        self.rel_exp_endowments = self.get_expected_endowments()
        print("R")
        print(self.rel_exp_endowments)



    def get_expected_endowments(self,N=1000):
        """
        Monte Carlo Method for estimating Expectation of type distribution using N realizations
        Only need to run this once to get expectations for all locations

        Returns: 
        rel_exp_endowments: matrix containing expected proportion of endowments for location t
        """
        num_types = self.env_config['weight_matrix'].shape[0]
        rel_exp_endowments = np.zeros((self.env_config['num_rounds'],num_types))
        for t in range(self.env_config['num_rounds']):
            mean_endowment = np.zeros(num_types)
            for i in range(N):
                endowment = self.env_config['type_dist'](t)
                mean_endowment += endowment   
            mean_endowment = (1/N)*mean_endowment
            rel_exp_endowments[t,:] = mean_endowment

        total = np.sum(rel_exp_endowments)
        for t in range(self.env_config['num_rounds']):
            for i in range(num_types):
                rel_exp_endowments[t,i] = rel_exp_endowments[t,i]/total
        
        
        assert np.sum(rel_exp_endowments) <= 1.5, "{}".format(rel_exp_endowments)
        return rel_exp_endowments

    def reset(self):
        # resets data matrix to be empty
        self.data = []
    
    def update_config(self, env, config):
        '''Updates environment configuration dictionary'''
        self.env_config = config
        return

    def update_obs(self, obs, action, reward, newObs, timestep, info): 
        '''Add observation to records'''
        self.data.append(newObs)
        return

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''
        num_types = self.env_config['weight_matrix'].shape[0]
        action = np.zeros((num_types, self.env_config['K']))
        print("timestep: %s"%timestep)
        for type in range(num_types):
            action[type,:] = self.env_config['init_budget']*self.rel_exp_endowments[timestep,type]
        
        return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
