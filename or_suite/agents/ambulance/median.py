import numpy as np
import agent
import sklearn_extra.cluster


''' Agent that implements a k-medoid heuristic algorithm for the line ambulance environment'''
class medianAgent(agent.FiniteHorizonAgent):

    def __init__(self, epLen):
        '''
        TODO: epLen - number of steps
        TODO: func - function used to decide action
        data - all data observed so far
        call_locs - the locations of all calls observed so far
        TODO: alpha - alpha parameter in ambulance problem
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

    def get_num_arms(self):
        return 0

    def update_policy(self, k):
        '''Update internal policy based upon records'''

        # Greedy algorithm does not update policy
        self.greedy = self.greedy


    def greedy(self, state, timestep, epsilon=0):
        '''
        Select action according to function
        '''

        # For the first iteration, choose the starting state
        # After that, choose locations for the ambulances that are most centrally
        # located to the locations of previous calls using the k-medoids algorithm
        # For more details about the k-medoids algorithm, see the readme document
        # for the ambulance environment or the sci-kit learn documentation
        if timestep == 0:
            return state
        else:
            num_ambulance = len(self.data[0])
            if len(self.call_locs) > num_ambulance:
                reshaped_call_locs = np.asarray(self.call_locs).reshape(-1,1)
                clusters = sklearn_extra.cluster.KMedoids(n_clusters=num_ambulance, max_iter=50).fit(reshaped_call_locs)
                action = np.asarray(clusters.cluster_centers_).reshape(-1,)
            else:
                action = np.full(num_ambulance, np.median(self.call_locs))
            return action


    def pick_action(self, state, step):
        action = self.greedy(state, step)
        return action
