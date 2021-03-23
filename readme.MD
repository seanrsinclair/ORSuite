   
# OR Suite
Reinforcement learning (RL) is a natural model for problems involving real-time sequential decision making. In these models, a principal interacts with a system having stochastic transitions and rewards and aims to control the system online (by exploring available actions using real-time feedback) or offline (by exploiting known properties of the system).

These project revolves around providing a unified landscape on scaling reinforcement learning algorithms to operations research domains.

### Installation Guide

In order to install the required dpeendencies for a new conda environment, please run:
```
conda create --name ORSuite python=3.8.5
conda activate ORSuite
python -m pip install -r requirements.txt
```

### High-Level Overview

The repository has three main components as a traditional Reinforcement Learning set-up :
1. Environments : Environment for the agent to interact with and reside in. `~/or_suite/envs`
2. Agents : Choice of Algorithm `~/or_suite/agents`
3. Experiments : This is a take on implementing the enviroment and agents with a choice of algorithm `~/or_suite/experiment`

All the three components are driven by a python script `~/simulations/run_<env_name>_save_data.py` which runs an experiment and generates the figures.


### Adding Environments

In order to add an additional environment to the ORSuite, simply:
1. Create Environment: Add an additional folder in `~/or_suite/envs/` corresponding to your problem set-up.  Then create a `env_name.py` file which inherits `gym.Env` dictating the MDP formulation.
2. Register Environment: Edit `~/or_suite/envs/__init__.py` to include a new entry defining your environment of the form
  ```python
    register(id = 'name-v0',
    entry_point = 'env_location'
    )
  ```
3. Set Configurations: Edit `~/or_suite/envs/env_configs.py` to include a configuration dictionary for default parameters of the environment.
4. Create an `__init__.py` file importing the new environment.

Note that `~/or_suite/envs/__init__.py` must also be editted to import the additional environment.

### Adding Algorithms

To add additional algorithms to the ORSuite:
1. If the algorithm is a *general purpose* RL algorithm, the file should be located under `~/or_suite/agents/rl/`, otherise, create a new subfolder under `~/or_suite/agents/` following the same name as the environment the algorithm is constructed for.
2. Create a new `name.py` file inside of that subfolder.  The algorithm should be a sub-class of `~/or_suite/agents/agent.py` dictating the naming convention and interaction structure that an algorithm must abide by.
3. Create an `__init__.py` file importing the new agent.

Note that `~/or_suite/agents/__init__.py` must be editted to import the additional algorithms.