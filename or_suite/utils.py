import numpy as np

def run_single_algo(env, agent, settings):

    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()





def mean_response_time(traj, dist):
    mrt = 0
    for i in range(len(traj)):
        cur_data = traj[i]
        mrt += np.min(dist(np.array(cur_data['action']),cur_data['info']['arrival']))
    return mrt / len(traj)