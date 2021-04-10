import numpy as np
import cvxpy as cp
import pandas as pd
import or_suite


'''

Helper code to run a single simulation of either an ORSuite experiment or the wrapper for a stable baselines algorithm.

'''

def run_single_algo(env, agent, settings):
    '''
        Runs a single experiment
        env - environment
        agent - agent
        setting - dictionary containing experiment settings
    '''
    exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()

def run_single_algo_tune(env, agent, scaling_list, settings):
    best_reward = (-1)*np.inf
    best_scaling = scaling_list[0]

    for scaling in scaling_list:
        agent.reset()
        agent.scaling = scaling

        exp = or_suite.experiment.experiment.Experiment(env, agent, settings)
        exp.run()
        dt = pd.DataFrame(exp.data, columns=['episode', 'iteration', 'epReward', 'memory', 'time'])
        avg_end_reward = dt[dt['episode'] == dt.max()['episode']].iloc[0]['epReward']
        if avg_end_reward >= best_reward:
            best_reward = avg_end_reward
            best_scaling = scaling_list[0]
            best_exp = exp
    best_exp.save_data()
    print(best_scaling)

# Helper code to run single stable baseline experiment

def run_single_sb_algo(env, agent, settings):
    '''
        Runs a single experiment
        env - environment
        agent - agent
        setting - dictionary containing experiment settings
    '''


    exp = or_suite.experiment.sb_experiment.SB_Experiment(env, agent, settings)
    _ = exp.run()
    dt_data = exp.save_data()



'''
PROBLEM DEPENDENT METRICS

Sample implementation of problem dependent metrics.  Each one of them should take in a trajectory (as output and saved in an experiment)
and return a corresponding value, where large corresponds to 'good'.

'''

# Calculating mean response time for ambulance environment on the trajectory datafile
def mean_response_time(traj, dist):
    mrt = 0
    for i in range(len(traj)):
        cur_data = traj[i]
        mrt += (-1)*np.min(dist(np.array(cur_data['action']),cur_data['info']['arrival']))
    return mrt / len(traj)

# Resoucre Allocation Metrics/Helper functions
def delta_OPT(traj, env_config):
    """
    Calculates the distance to X_opt w.r.t supremum norm
    
    Inputs:
        traj: trajectory of an algorithm, stored as a list of dictionaries
        env_config: configuration of the environment
    Returns:
        final_avg_dist: array of the average dist to X_opt for each episode

    """
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    #print('Iters: %s, Eps: %s, Steps: %s'%(num_iter,num_eps,num_steps))
    num_types,num_commodities = traj[-1]['action'].shape 
    final_avg_dist = np.zeros(num_eps)
    
    for iteration in range(num_iter):      
        iteration_traj = list(filter(lambda d: d['iter']==iteration, traj))
        
        for ep in range(num_eps):
            ep_traj = list(filter(lambda d: d['episode']==ep, traj))
            sizes = np.zeros((num_steps,num_types))

            for idx,step_dict in enumerate(ep_traj):
                size = step_dict['info']['type']
                sizes[idx,:] = size
                   
            prob, solver = generate_cvxpy_solve(num_types,num_commodities)
            X_opt = offline_opt(env_config['init_budget'],sizes,env_config['weight_matrix'],solver)
            X_alg = np.zeros((num_steps,num_types,num_commodities))
            
            for idx,step_dict in enumerate(ep_traj):
                X_alg[idx,:,:] = step_dict['action']
            
            dist = np.max(np.absolute(X_opt-X_alg))
            final_avg_dist[ep] += (1/num_iter)*dist
            #print("Dist to OPT for episode %s: %s"%(ep,dist))
            
    return (-1)*np.mean(final_avg_dist)


def delta_proportionality(traj, env_config):
    """
    Calculate the proportionality (distance to equal allocation) at each episode
    
    Inputs:
        traj: trajectory of an algorithm, stored as a list of dictionaries
        env_config: configuration of the environment
    Returns:
        final_avg_efficiency: array containing average waste per episode 

    """
    
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    #print('Iters: %s, Eps: %s, Steps: %s'%(num_iter,num_eps,num_steps))
    num_types,num_commodities = traj[-1]['action'].shape 
    final_avg_efficiency = np.zeros(num_eps)
    
    for iteration in range(num_iter):      
        iteration_traj = list(filter(lambda d: d['iter']==iteration, traj))

        for ep in range(num_eps):

            ep_traj = list(filter(lambda d: d['episode']==ep, traj))
            sizes = np.zeros((num_steps,num_types))

            for idx,step_dict in enumerate(ep_traj):
                size = step_dict['info']['type']
                sizes[idx,:] = size

            X_alg = np.zeros((num_steps,num_types,num_commodities))
            
            for idx,step_dict in enumerate(ep_traj):
                X_alg[idx,:,:] = step_dict['action']
            
            prop = get_proportionality(X_alg,sizes,env_config)
            final_avg_efficiency[ep] += (1/num_iter)*prop
            #print("Proportionality for episode %s: %s"%(ep,prop))
            
    return (-1)*np.mean(final_avg_efficiency)


def delta_efficiency(traj, env_config):
    """
    Calculate the efficiency (waste) of an algorithm given its trajectory
    
    Inputs:
        traj: trajectory of an algorithm, stored as a list of dictionaries
        env_config: configuration of the environment
    Returns:
        final_avg_efficiency: array containing average waste per episode 

    """
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    #print('Iters: %s, Eps: %s, Steps: %s'%(num_iter,num_eps,num_steps))
    num_types,num_commodities = traj[-1]['action'].shape 
    final_avg_efficiency = np.zeros(num_eps)
    for iteration in range(num_iter):      
        iteration_traj = list(filter(lambda d: d['iter']==iteration, traj))

        for ep in range(num_eps):
            ep_traj = list(filter(lambda d: d['episode']==ep, traj))
            sizes = np.zeros((num_steps,num_types))

            for idx,step_dict in enumerate(ep_traj):
                size = step_dict['info']['type']
                sizes[idx,:] = size
            
            X_alg = np.zeros((num_steps,num_types,num_commodities))
            
            for idx,step_dict in enumerate(ep_traj):
                X_alg[idx,:,:] = step_dict['action']
            
            eff = get_efficiency(X_alg,sizes,env_config)
            final_avg_efficiency[ep] += (1/num_iter)*eff
            #print("Efficiency for episode %s: %s"%(ep,eff))

    return (-1)*np.mean(final_avg_efficiency)


def delta_envy(traj, env_config):
    """
    Calculates the delta_envy metric given the trajectory of a given algorithm
    
    Inputs:
        traj: trajectory of an algorithm, stored as a list of dictionaries
        env_config: configuration of the environment
    Returns:
        final_avg_envies: array of the average envy for each episode

    """
    num_iter = traj[-1]['iter']+1
    num_eps = traj[-1]['episode']+1
    num_steps = traj[-1]['step']+1
    #print('Iters: %s, Eps: %s, Steps: %s'%(num_iter,num_eps,num_steps))
    num_types,num_commodities = traj[-1]['action'].shape 
    final_avg_envies = np.zeros(num_eps)
    
    for iteration in range(num_iter):      
        iteration_traj = list(filter(lambda d: d['iter']==iteration, traj))
        
        for ep in range(num_eps):
            ep_traj = list(filter(lambda d: d['episode']==ep, traj))
            sizes = np.zeros((num_steps,num_types))

            for idx,step_dict in enumerate(ep_traj):
                size = step_dict['info']['type']
                sizes[idx,:] = size
                   
            prob, solver = generate_cvxpy_solve(num_types,num_commodities)
            X_opt = offline_opt(env_config['init_budget'],sizes,env_config['weight_matrix'],solver)
            X_alg = np.zeros((num_steps,num_types,num_commodities))
            
            for idx,step_dict in enumerate(ep_traj):
                X_alg[idx,:,:] = step_dict['action']
            
            envy = get_envy(X_alg,X_opt,env_config)
            final_avg_envies[ep] += (1/num_iter)*envy
            #print("Envy for episode %s: %s"%(ep,envy))
            
    return (-1)*np.mean(final_avg_envies)


def offline_opt(budget, size, weights, solver):
    """
    Uses solver from generate_cvxpy_solve and applies it to values
    
    Inputs:
        budget: initial budget for K commodities
        size: 2D numpy array of sizes of each type at each location
        weights: 2D numpy array containing the demands of each type
    """
    tot_size = np.sum(size, axis=0)
    _, x = solver(tot_size, weights, budget)
    allocation = np.zeros((size.shape[0], weights.shape[0], weights.shape[1]))
    for i in range(size.shape[0]):
        allocation[i,:,:] = x
    return allocation


#SEANS CODE FOR GENERATING SOLVER FOR OFFLINE PROBLEM
def generate_cvxpy_solve(num_types, num_resources):
    """
    Creates a generic solver to solve the offline resource allocation problem
    
    Inputs: 
        num_types - number of types
        num_resources - number of resources
    Returns:
        prob - CVXPY problem object
        solver - function that solves the problem given data
    """
    x = cp.Variable(shape=(num_types,num_resources))
    sizes = cp.Parameter(num_types, nonneg=True)
    weights = cp.Parameter((num_types, num_resources), nonneg=True)
    budget = cp.Parameter(num_resources, nonneg=True)
    objective = cp.Maximize(cp.log(cp.sum(cp.multiply(x, weights), axis=1)) @ sizes)
    constraints = []
    constraints += [0 <= x]
    for i in range(num_resources):
        constraints += [x[:, i] @ sizes <= budget[i]]
    # constraints += [x @ sizes <= budget]
    prob = cp.Problem(objective, constraints)
    def solver(true_sizes, true_weights, true_budget):
        sizes.value = true_sizes
        weights.value = true_weights
        budget.value = true_budget
        prob.solve()
        return prob.value, np.around(x.value, 5)
    return prob, solver


def get_proportionality(X_alg,sizes,env_config):
    """
    (helper for delta_proportionality)
    Finds proportionality by calculating envy w.r.t a completely equal allocation
    """
    B = env_config['init_budget']
    tot_size = np.sum(sizes)
    u = env_config['utility_function']
    w = env_config['weight_matrix']
    max_prop=0
    for t,allocation in enumerate(X_alg):
        for theta,row in enumerate(allocation):
            tmp = abs(u(row,w[theta,:])-u(B/tot_size,w[theta,:]))
            if tmp >= max_prop:
                max_prop = tmp
    return max_prop


def get_efficiency(X_alg, sizes,env_config):
    """
    (helper for delta_efficiency)
    Finds efficiency by seeing how much of the initial budget was used in X_alg
    """
    B = env_config['init_budget']
    tot_sizes = np.sum(sizes, axis=0)
    num_types,num_commodities = env_config['weight_matrix'].shape
    return sum([B-sum([tot_sizes[theta]*X_alg[t][theta,:] for theta in range(num_types)]) for t in range(len(X_alg))])


def get_envy(X_alg,X_opt,env_config):
    """
    (helper for delta_envy)
    Finds maximum envy of X_alg's allocation by comparing its utility to that of X_opt
    """
    u = env_config['utility_function']
    w = env_config['weight_matrix']
    max_envy=0
    for t,allocation in enumerate(X_alg):
        for theta,row in enumerate(allocation):
            tmp = abs(u(row,w[theta,:])-u(X_opt[t][theta,:],w[theta,:]))
            if tmp >= max_envy:
                max_envy = tmp
    return max_envy