from .. import Agent

class randomAgent(Agent):

    def __init__(self):
        pass

    def update_config(self, config):
        ''' Update agent information based on the config__file'''
        pass

    def update_obs(self, obs, action, reward, newObs):
        pass

    def update_policy(self, h):
        pass

    def pick_action(self, obs):
        '''Select an action based upon the observation'''
        pass