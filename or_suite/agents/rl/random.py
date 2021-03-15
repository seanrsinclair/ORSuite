from .. import Agent

class randomAgent(Agent):

    def __init__(self):
        pass


    def reset(self):
        pass

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.environment = env
        pass

    def update_obs(self, obs, action, reward, newObs, timestep, info):
        pass

    def update_policy(self, h):
        pass

    def pick_action(self, obs, h):
        '''Select an action based upon the observation'''
        return self.environment.action_space.sample()