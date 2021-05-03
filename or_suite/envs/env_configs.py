'''

File containing default configurations for the various environments implemented in ORSuite.

'''

import numpy as np

resource_allocation_default_config = {'K': 2, 
    'num_rounds': 10,
    'weight_matrix': np.array([[1,2],[.3,9],[1,1]]),
    'init_budget': 10*np.ones(2),
    'type_dist': lambda i: 1+np.random.poisson(size=(3), lam = (1,2,3)),
    'utility_function': lambda x,theta: np.dot(x,theta)
    }

resource_allocation_simple_config = {'K':1,
  'num_rounds':10,
  'weight_matrix': np.array([[1]]),
  'init_budget': np.array([20.]),
  'utility_function': lambda x,theta: x,
  'type_dist': lambda i : np.array([2])
}

resource_allocation_simple_poisson_config = {'K':1,
  'num_rounds':10,
  'weight_matrix': np.array([[1]]),
  'init_budget': np.array([20.]),
  'utility_function': lambda x,theta: x,
  'type_dist': lambda i : [1+np.random.poisson(lam = 1)]
}


ambulance_metric_default_config =  {'epLen': 5,
    'arrival_dist': lambda x : np.random.beta(5,2), 
    'alpha': 0.25, 
    'starting_state': np.array([0.0]), 
    'num_ambulance': 1
  }


ambulance_graph_default_config = {'epLen': 5, 
    'arrival_dist': lambda step, num_nodes: np.full(num_nodes, 1/num_nodes), 
    'alpha': 0.25,
    'from_data': False,
    'edges': [(0,4,{'travel_time':7}), (0,1,{'travel_time':1}), (1,2,{'travel_time':3}), (2,3,{'travel_time':5}), (1,3,{'travel_time':1}), (1,4,{'travel_time':17}), (3,4,{'travel_time':3})],
                'starting_state': [1,2], 'num_ambulance': 2
  }


finite_bandit_default_config =  {'epLen': 5,
    'arm_means': np.array([.1, .7, .2, 1])
  }

vaccine_4groups_default_config = {'epLen': 4, 
    'starting_state': np.array([990, 1990, 990, 5990, 10, 10, 10, 10, 0, 0, 0]), 
    'parameters': {'contact_matrix':np.array([[0.0001, 0.0001, 0.00003, 0.00003],[0, 0.0001, 0.00005, 0.0001],[0, 0, 0.00003, 0.00003],[0, 0, 0, 0.00003]]), 
                   'lambda_hosp': 0.0001,
                   'rec': 0,
                   'p1': 0.15, 'p2': 0.15, 'p3': 0.7, 'p4': 0.2,
                   'h1': 0.2, 'h2': 0.2, 'h3': 0.7,'h4': 0.3, 
                   'gamma': 100, 
                   'beta': 1/7, 
                   'priority_order': [],
                   'vaccines': 625, 
                   'time_step':7}
  }

rideshare_graph_default_config = {
    'epLen': 5,
    'edges': [(0,1,{'travel_time':1}), (0,2,{'travel_time':1}), 
            (0,3,{'travel_time':1}), (1,2,{'travel_time':1}), 
            (1,3,{'travel_time':1}), (2,3,{'travel_time':1})], 
    'starting_state': [2,1,4,3], 
    'num_cars': 10,
    'request_dist': lambda step, num_nodes: np.random.choice(num_nodes, size=2),
    'reward': lambda distance: -np.sqrt(distance),
    'reward_fail': lambda distance: -10000,
    'gamma': 1
}


oil_environment_default_config = {
    'epLen': 5,
    'dim': 1,
    'starting_state' : np.asarray([0]),
    'oil_prob': lambda x,a,h : np.exp((-1)*np.sum(np.abs(x-a))),
    'cost_param' : 0,
    'noise_variance' : lambda x,a,h : 0

}
