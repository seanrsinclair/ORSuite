import numpy as np

resource_allocation_default_cofig = {'K': 2, 
    'num_rounds': 3,
    'weight_matrix': np.array([[1,0],[0,1],[1,1]]),
    'init_budget': 100*np.ones(2),
    'type_dist': lambda i: np.random.randint(50,size=3),
    'utility_function': lambda x,theta: np.dot(x,theta)
    }



ambulance_metric_default_config =  {'epLen': 5,
    'arrival_dist': lambda x : np.random.beta(5,2), 
    'alpha': 0.25, 
    'starting_state': np.array([0.0]), 
    'num_ambulance': 1
  }


ambulance_graph_default_config = {'epLen': 5, 
    'arrival_dist': None, 
    'alpha': 0.25,
    'edges': [(0,4,{'dist':7}), (0,1,{'dist':1}), (1,2,{'dist':3}), (2,3,{'dist':5}), (1,3,{'dist':1}), (1,4,{'dist':17}), (3,4,{'dist':3})],
                'starting_state': [1,2], 'num_ambulance': 2
  }


finite_bandit_default_config =  {'epLen': 5,
    'arm_means': np.array([.1, .7, .2, 1])
  }


