from .. import Agent

class randomAgent(Agent):

    def __init__(self):
        pass

<<<<<<< HEAD

    def reset(self):
        pass

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        self.environment = env
        pass

    def update_obs(self, obs, action, reward, newObs, timestep, info):
=======
    def update_config(self, config):
        ''' Update agent information based on the config__file'''
        pass

    def update_obs(self, obs, action, reward, newObs):
>>>>>>> 72b65ac (Line figures and plots)
        pass

    def update_policy(self, h):
        pass

<<<<<<< HEAD
    def pick_action(self, obs, h):
        '''Select an action based upon the observation'''
        return self.environment.action_space.sample()
=======
    def pick_action(self, obs):
        '''Select an action based upon the observation'''
        pass
>>>>>>> 72b65ac (Line figures and plots)
