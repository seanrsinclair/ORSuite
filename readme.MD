   
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

The repository has three main components as a traditional Reinforcement Learning set-up :
1. Environments : Environment for the agent to interact with and reside in. `~/or_suite-main/or-suite/envs`
2. Agents : Choice of Algorithm `~/or_suite-main/or-suite/agents`
3. Experiments : This is a take on implementing the enviroment and agents with a choice of algorithm `~/or_suite-main/or-suite/experiments`

All the three components are driven by a python script `run_<env_name>_save_data.py`.

### Quick Start Guide
- Step 1 : Execute the following command to generate data points for *__ambulance__* environment: 
` python run_ambulance_save_data.py`
This saves the data generated from the *__ambulance__* environment to `~/or_suite-main/data` 
The file structure will be in the format of `~/or_suite-main/data/<env_name>_metric_<agent>_<sweep function>/data.csv`
The generated data will contain the following metrics : 
  - Episodes Rewards
  - Iteration counts
  - Time taken per episode
  - Memory usage per episode
- Step 2 : Generating Plots using the generated data files. :
 `python create_fig.py`
The resulting plots will be of two types :
  - Line plots comparing the agents (algorithms)
  - Radar plots (scaling of variables required)
- This will create a figures folder in `~/or_suite-main/figures/`
