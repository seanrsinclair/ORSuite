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
  'init_budget': np.array([10.]),
  'utility_function': lambda x,theta: x,
  'type_dist': lambda i : np.array([2])
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


