from gym.envs.registration import register
import or_suite.envs.ambulance
import or_suite.envs.resource_allocation
import or_suite.envs.finite_armed_bandit
import or_suite.envs.vaccine_allotment

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

# Finite Armed Bandit

register(id = 'Bandit-v0',
    entry_point = 'or_suite.envs.finite_armed_bandit.finite_bandit:FiniteBanditEnvironment'
)

# Vaccine Allotment Environments

register(id = 'Vaccine-v0',
         entry_point = 'or_suite.envs.vaccine_allotment.vacc_4groups:VaccineEnvironment'
)
