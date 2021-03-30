import numpy as np
import or_suite


# Helper code to run experiment for single algorithm

def run_single_algo(env, agent, settings):

    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()

# Helper code to run single stable baseline experiment

def run_single_sb_algo(env, agent, settings):

    exp = or_suite.experiment.sb_experiment.SB_Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()


# Calculating mean response time for ambulance environment on the trajectory datafile

def mean_response_time(traj, dist):
    mrt = 0
    for i in range(len(traj)):
        cur_data = traj[i]
        mrt += (-1)*np.min(dist(np.array(cur_data['action']),cur_data['info']['arrival']))
    return mrt / len(traj)