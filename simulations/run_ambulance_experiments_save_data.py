import sys
sys.path.insert(1, '../')
import numpy as np
import gym
from adaptive_Agent import AdaptiveDiscretization
from eNet_model_Agent import eNetModelBased
from eNet_Agent import eNet
from adaptive_model_Agent import AdaptiveModelBasedDiscretization
from data_Agent import dataUpdateAgent
from src import environment
from src import experiment
from src import agent
import pickle
import multiprocessing as mp
from joblib import Parallel, delayed
import time


def run_single_algo(algorithm, env, dictionary, path, num_iters, epLen, nEps):


    # Running Experiments

    exp = experiment.Experiment(env, agent_list, dictionary)
    _ = exp.run()
    dt_data = exp.save_data()


    # Saving Data
    dt_final_data.to_csv(path+'.csv')
    agent = opt_agent_list[-1]
    filehandler = open(path+'.obj', 'wb')
    pickle.dump(agent, filehandler)

    return (algorithm,opt_param,opt_reward)

''' Defining parameters to be used in the experiment'''



ambulance_list = ['shifting', 'beta', 'uniform']
param_list_ambulance = ['0', '1', '25']

#TODO: Edit algo-list to be the names of the algorithms you created
algo_list = ['adaMB_One', 'adaMB_Full', 'adaMB_One_Flag', 'adaMB_Full_Flag', 'adaMB_One_3', 'adaMB_Full_3', 'adaMB_One_Flag_3', 'adaMB_Full_Flag_3', 'adaQL', 'epsMB_One', 'epsMB_Full', 'epsQL']


for problem in ambulance_list:
    for param in param_list_ambulance:

        epLen = 5
        nEps = 2000
        numIters = 200
        if problem == 'beta':
            def arrivals(step):
                return np.random.beta(5,2)
        elif problem == 'uniform':
            def arrivals(step):
                return np.random.uniform(0,1)
        elif problem == 'shifting':
            def arrivals(step):
                if step == 0:
                    return np.random.uniform(0, .25)
                elif step == 1:
                    return np.random.uniform(.25, .3)
                elif step == 2:
                    return np.random.uniform(.3, .5)
                elif step == 3:
                    return np.random.uniform(.5, .6)
                else:
                    return np.random.uniform(.6, .65)
        if param == '1':
            alpha = 1
        elif param == '0':
            alpha = 0
        else:
            alpha = 0.25

        starting_state = 0.5

        #TODO: Update to create your version of the ambulance.  Can just pick one dimension ambulance, but make sure you set
        # arrival distribution and cost parameter as defined above.

        env = environment.make_ambulanceEnvMDP(epLen, arrivals, alpha, starting_state)

        ##### PARAMETER TUNING FOR AMBULANCE ENVIRONMENT

        scaling_list = [0.01, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 5]

        path = {}
        for algorithm in algo_list:
            path[algorithm] = '../data/ambulance_'+problem+'_'+param+'_'+algorithm
            run_single_algo(algorithm, env, dictionary, path[algorithm], numIters, epLen, nEps)
