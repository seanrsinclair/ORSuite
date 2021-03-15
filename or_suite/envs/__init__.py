from gym.envs.registration import register
import or_suite.envs.ambulance
import or_suite.envs.resource_allocation
from or_suite.envs.env_configs import *


# Ambulance Environments

register(id = 'Ambulance-v0',
    entry_point ='or_suite.envs.ambulance.ambulance_metric:AmbulanceEnvironment'
)

register(id = 'Ambulance-v1',
    entry_point ='or_suite.envs.ambulance.ambulance_graph:AmbulanceGraphEnvironment'
)

# Resource Allocation Environments

register(id = 'Resource-v0',
    entry_point = 'or_suite.envs.resource_allocation.resource_allocation:ResourceAllocationEnvironment'
)
