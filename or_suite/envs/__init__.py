from gym.envs.registration import register
import or_suite.envs.ambulance
import or_suite.envs.resource_allocation
<<<<<<< HEAD
import or_suite.envs.finite_armed_bandit

from or_suite.envs.env_configs import *
=======
>>>>>>> 72b65ac (Line figures and plots)


# Ambulance Environments

register(id = 'Ambulance-v0',
    entry_point ='or_suite.envs.ambulance.ambulance_metric:AmbulanceEnvironment'
)

register(id = 'Ambulance-v1',
    entry_point ='or_suite.envs.ambulance.ambulance_graph:AmbulanceGraphEnvironment'
)

# Resource Allocation Environments

register(id = 'Resource-v0',
<<<<<<< HEAD
    entry_point = 'or_suite.envs.resource_allocation.resource_allocation:ResourceAllocationEnvironment'
)

# Finite Armed Bandit
register(id = 'Bandit-v0',
    entry_point = 'or_suite.envs.finite_armed_bandit.finite_bandit:FiniteBanditEnvironment'
=======
    entry_point = 'or_suite.envs.resource_allocation:resource_allocation:ResourceAllocationEnvironment'
>>>>>>> 72b65ac (Line figures and plots)
)
