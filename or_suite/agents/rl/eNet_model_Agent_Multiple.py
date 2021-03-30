import numpy as np
from .. import Agent
import itertools

class eNetModelBased(Agent):

    def __init__(self, action_net, state_net, epLen, scaling, state_action_dim, alpha, flag):
        '''
        args:
            - action_net - epsilon net of action space
            - state_net - epsilon net of state space
            - epLen - steps per episode
            - scaling - scaling parameter for UCB terms
        '''

        self.state_net = np.resize(state_net, (state_action_dim[0], len(state_net))).T
        self.action_net = np.resize(action_net, (state_action_dim[1], len(action_net))).T
        self.epLen = epLen
        self.scaling = scaling
        self.alpha = alpha
        self.flag = flag
        self.state_action_dim = state_action_dim
        self.state_size = self.state_action_dim[0] * [len(state_net)]
        self.action_size = self.state_action_dim[1] * [len(action_net)]
        self.qVals = np.ones([self.epLen]+self.state_size+self.action_size, dtype=np.float32) * self.epLen
        self.num_visits = np.zeros([self.epLen]+ self.state_size+self.action_size, dtype=np.float32)
        self.vVals = np.ones([self.epLen]+ self.state_size, dtype=np.float32) * self.epLen
        self.rEst = np.zeros([self.epLen]+ self.state_size+ self.action_size, dtype=np.float32)
        self.pEst = np.zeros([self.epLen]+ self.state_size+ self.action_size+self.state_size,
                             dtype=np.float32)

        '''
            Resets the agent by overwriting all of the estimates back to zero
        '''
    def reset(self):
        self.qVals = np.ones([self.epLen]+ self.state_size+ self.action_size, dtype=np.float32) * self.epLen
        self.vVals = np.ones([self.epLen]+ self.state_size, dtype=np.float32) * self.epLen
        self.rEst = np.zeros([self.epLen]+ self.state_size+ self.action_size, dtype=np.float32)
        self.num_visits = np.zeros([self.epLen]+ self.state_size+ self.action_size, dtype=np.float32)
        self.pEst = np.zeros([self.epLen]+ self.state_size+ self.action_size+self.state_size,
                             dtype=np.float32)
        '''
            Adds the observation to records by using the update formula
        '''
    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''

        # returns the discretized state and action location
        state_discrete = np.argmin((np.abs(self.state_net - np.asarray(obs))), axis=0)
        action_discrete = np.argmin((np.abs(self.action_net - np.asarray(action))), axis=0)
        state_new_discrete = np.argmin((np.abs(self.state_net - np.asarray(newObs))), axis=0)

        dim = (timestep,) + tuple(state_discrete) + tuple(action_discrete)
        self.num_visits[dim] += 1

        self.pEst[dim+tuple(state_new_discrete)] += 1
        t = self.num_visits[dim]
        self.rEst[dim] = ((t - 1) * self.rEst[dim] + reward) / t



    def get_num_arms(self):
        ''' Returns the number of arms'''
        return self.epLen * len(self.state_net)**(self.state_action_dim[0]) * len(self.action_net)**(self.state_action_dim[1])

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # Update value estimates
        if self.flag:
            for h in np.arange(self.epLen - 1, -1, -1):
                for state in itertools.product(*[np.arange(len(self.state_net)) for _ in range(self.state_action_dim[0])]):
                    for action in itertools.product(*[np.arange(len(self.action_net)) for _ in range(self.state_action_dim[1])]):
                        dim = (h,) + state + action
                        if self.num_visits[dim] == 0:
                            self.qVals[dim] = self.epLen
                        else:
                            if h == self.epLen - 1:
                                self.qVals[dim] = min(self.qVals[dim], self.rEst[dim] + self.scaling / np.sqrt(self.num_visits[dim]))
                            else:
                                vEst = min(self.epLen, np.sum(np.multiply(self.vVals[(h+1,)], self.pEst[dim] + self.alpha) / (np.sum(self.pEst[dim] + self.alpha))))
                                self.qVals[dim] = min(self.qVals[dim], self.epLen, self.rEst[dim] + self.scaling / np.sqrt(self.num_visits[dim]) + vEst)
                    self.vVals[(h,) + state] = min(self.epLen, self.qVals[(h,) + state].max())

        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # returns the discretized state location and takes action based on
        # maximum q value
        state_discrete = np.argmin((np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
        qFn = self.qVals[(timestep,)+tuple(state_discrete)]
        action = np.asarray(np.where(qFn == qFn.max()))
        a = len(action[0])
        index = np.random.choice(len(action[0]))

        actions = ()
        for val in action.T[index]:
            actions += (self.action_net[:,0][val],)
        return actions

    def pick_action(self, state, step):
        if self.flag == False:
            state_discrete = np.argmin((np.abs(np.asarray(self.state_net) - np.asarray(state))), axis=0)
            for action in itertools.product(*[np.arange(len(self.action_net)) for _ in range(self.state_action_dim[1])]):
                dim = (step,) + tuple(state_discrete) + action
                if self.num_visits[dim] == 0:
                    self.qVals[dim] == 0
                else:
                    if step == self.epLen - 1:
                        self.qVals[dim] = min(self.qVals[dim], self.rEst[dim] + self.scaling / np.sqrt(self.num_visits[dim]))
                    else:
                        vEst = min(self.epLen, np.sum(np.multiply(self.vVals[(step+1,)], self.pEst[dim] + self.alpha) / (np.sum(self.pEst[dim] + self.alpha))))
                        self.qVals[dim] = min(self.qVals[dim], self.epLen, self.rEst[dim] + self.scaling / np.sqrt(self.num_visits[dim]) + vEst)
            self.vVals[(step,)+tuple(state_discrete)] = min(self.epLen, self.qVals[(step,) + tuple(state_discrete)].max())

        action = self.greedy(state, step)
        return action
