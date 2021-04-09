import numpy as np
from .. import Agent
from or_suite.agents.rl.utils.tree import Tree, Node

class AdaptiveDiscretization(Agent):

    def __init__(self, epLen, scaling, inherit_flag, dim):
        '''args:
            epLen - number of steps per episode
            numIters - total number of iterations
            scaling - scaling parameter for UCB term
        '''
        self.epLen = epLen
        self.scaling = scaling
        self.inherit_flag = inherit_flag
        self.dim = dim

        # List of tree's, one for each step
        self.tree_list = []

        # Makes a new partition for each step and adds it to the list of trees
        for _ in range(epLen):
            tree = Tree(epLen, self.dim)
            self.tree_list.append(tree)

    def reset(self):
        # Resets the agent by setting all parameters back to zero
        self.tree_list = []
        for _ in range(self.epLen):
            tree = Tree(self.epLen, self.dim)
            self.tree_list.append(tree)

    def update_config(self, env, config):
        ''' Update agent information based on the config__file'''
        pass

    
    # Gets the number of arms for each tree and adds them together
    def get_num_arms(self):
        total_size = 0
        for tree in self.tree_list:
            total_size += tree.get_number_of_active_balls()
        return total_size



    def update_obs(self, obs, action, reward, newObs, timestep, info):
        '''Add observation to records'''
        # Gets the active tree based on current timestep
        # print(obs, action, newObs)
        tree = self.tree_list[timestep]

        # Gets the active ball by finding the argmax of Q values of relevant
        active_node, _ = tree.get_active_ball(obs)

        if timestep == self.epLen - 1:
            vFn = 0

        else:
            # Gets the next tree to get the approximation to the value function
            # at the next timestep
            new_tree = self.tree_list[timestep + 1]
            _, new_q = new_tree.get_active_ball(newObs)

            vFn = min(self.epLen, new_q)


        # Updates parameters for the node
        active_node.num_visits += 1
        t = active_node.num_visits
        
        lr = (self.epLen + 1) / (self.epLen + t)
        bonus = self.scaling * np.sqrt(1 / t)
        
        active_node.qVal = (1 - lr) * active_node.qVal + lr * (reward + vFn + bonus)

        '''determines if it is time to split the current ball'''
        if t >= 2**(2*active_node.depth):
            active_node.split_node(self.inherit_flag)


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to a greedy policy

        Args:
            state - int - current state
            timestep - int - timestep *within* episode

        Returns:
            action - int
        '''
        # Considers the partition of the space for the current timestep
        tree = self.tree_list[timestep]

        # Gets the selected ball
        active_node, _ = tree.get_active_ball(state)

        # Picks an action uniformly in that ball
        # action = np.random.uniform(active_node.action_val - active_node.radius, active_node.action_val + active_node.radius)
        action_dim = self.dim - len(state)
        action = np.random.uniform(active_node.bounds[action_dim:, 0], active_node.bounds[action_dim:, 1])
        return action

    def update_policy(self, k):
        '''Update internal policy based upon records'''
        # TODO: Verify this is needed.
        # self.greedy = self.greedy
        return


    def pick_action(self, state, timestep):
        action = self.greedy(state, timestep)
        return action
