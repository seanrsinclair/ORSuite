from gym.envs.registration import register
import or_suite.envs.ambulance
import or_suite.envs.resource_allocation


# Ambulance Environments

register(id = 'Ambulance-v0',
    entry_point ='or_suite.envs.ambulance.ambulance_metric:AmbulanceEnv'
)

register(id = 'Ambulance-v1',
    entry_point ='or_suite.envs.ambulance.ambulance_graph:AmbulanceEnv'
)

# Resource Allocation Environments

register(id = 'Resource-v0',
    entry_point = 'or_suite.envs.resource_allocation:resource_allocation:ResourceAllocationEnvironment'
)
