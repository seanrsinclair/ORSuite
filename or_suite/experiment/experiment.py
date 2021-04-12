import time
from shutil import copyfile
import pandas as pd
import tracemalloc
import numpy as np
import pickle
import os

class Experiment(object):

    def __init__(self, env, agent, dict):
        '''
        A simple class to run a MDP Experiment.
        Args:
            env - an instance of an Environment
            agent - an agent
            dict - a dictionary containing the arguments to send for the experiment, including:
                seed - random seed for experiment
                recFreq - proportion of episodes to save to file
                dirPath - path to the file for saving
                deBug - boolean of whether to include
                nEps - number of episodes
                numIters - the number of iterations to run experiment
                saveTrajectory - boolean of whether to save trajectory information
        '''

        self.seed = dict['seed']
        self.epFreq = dict['recFreq']
        self.dirPath = dict['dirPath']
        self.deBug = dict['deBug']
        self.nEps = dict['nEps']
        self.env = env
        self.epLen = dict['epLen']
        self.num_iters = dict['numIters']
        self.save_trajectory = dict['saveTrajectory']
        self.agent = agent
        self.data = np.zeros([dict['nEps']*self.num_iters, 5])


        if self.save_trajectory:
            self.trajectory = []

        np.random.seed(self.seed)

    # Runs the experiment
    def run(self):
        print('**************************************************')
        print('Running experiment')
        print('**************************************************')


        index = 0
        traj_index = 0
        for i in range(self.num_iters):
            self.agent.reset() # resets algorithm, updates based on environment's configuration
            self.agent.update_config(self.env, self.env.get_config())
            for ep in range(0, self.nEps):
                if self.deBug:
                    print('Episode : ' + str(ep))
                # Reset the environment
                self.env.reset()

                self.env.render()
                time.sleep(2)

                oldState = self.env.state
                epReward = 0

                self.agent.update_policy(ep)

                done = False
                h = 0

                start_time = time.time()
                tracemalloc.start() # starts time and memory tracker

                # repeats until episode is finished
                while (not done) and h < self.epLen:
                    # Step through the episode
                    if self.deBug:
                        print('state : ' + str(oldState))
                    action = self.agent.pick_action(oldState, h)
                    if self.deBug:
                        print('action : ' + str(action))

                    newState, reward, done, info = self.env.step(action)
                    epReward += reward

                    if self.deBug:
                        print('new state: ' + str(newState))
                        print('reward: ' + str(reward))
                        print('epReward so far: ' + str(epReward))

                    self.agent.update_obs(oldState, action, reward, newState, h, info)

                    if self.save_trajectory: # saves trajectory step is desired
                        record = {'iter': i, 'episode': ep, 'step' : h, 'oldState' : oldState, 'action' : action, 'reward' : reward, 'newState' : newState, 'info' : info}
                        self.trajectory.append(record)

                    oldState = newState
                    h = h + 1

                    self.env.render()
                    time.sleep(2)

                current, peak = tracemalloc.get_traced_memory() # collects memory / time usage
                tracemalloc.stop()
                end_time = time.time()
                
                if self.deBug:
                    print('final state: ' + str(newState))


                # Logging to dataframe
                self.data[index, 0] = ep
                self.data[index, 1] = i
                self.data[index, 2] = epReward
                self.data[index, 3] = current
                self.data[index, 4] = np.log(((end_time) - (start_time)))

                index += 1

            self.env.close()

        print('**************************************************')
        print('Experiment complete')
        print('**************************************************')

    # Saves the data to the file location provided to the algorithm
    def save_data(self): 
        print('**************************************************')
        print('Saving data')
        print('**************************************************')

        print(self.data)

        dir_path = self.dirPath

        data_loc = 'data.csv'
        traj_loc = 'trajectory.obj'


        # Determines if we are saving the trajectory
        if self.save_trajectory:

            dt = pd.DataFrame(self.data, columns=['episode', 'iteration', 'epReward', 'memory', 'time'])
            dt = dt[(dt.T != 0).any()]

            filename = os.path.join(dir_path, traj_loc)

            print('Writing to file ' + data_loc)
        else:

            dt = pd.DataFrame(self.data, columns=['episode', 'iteration', 'epReward', 'memory', 'time'])
            dt = dt[(dt.T != 0).any()]
            print('Writing to file ' + data_loc)

        if os.path.exists(dir_path):
            # saves the collected dataset
            dt.to_csv(os.path.join(dir_path,data_loc), index=False, float_format='%.5f', mode='w')
            if self.save_trajectory: # saves trajectory to filename
                outfile = open(filename, 'wb')
                pickle.dump(self.trajectory, outfile)
                outfile.close()
        else: # same as before, but first makes the directory
            os.makedirs(dir_path)
            dt.to_csv(os.path.join(dir_path, data_loc), index=False, float_format='%.5f', mode='w')
            if self.save_trajectory:
                outfile = open(filename, 'wb')
                pickle.dump(self.trajectory, outfile)
                outfile.close()

        print('**************************************************')
        print('Data save complete')
        print('**************************************************')

        return dt