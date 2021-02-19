## Overview

### Description

`ambulance_graph.py` is a reinforcement learning environment meant to represent a situation where ambulances can be stationed at different locations, and the nearest one will respond to an incoming call. Ambulances can also move to new locations between calls. The environment is structured as a graph of nodes where each node represents a location where a call for an ambulance could come in. The edges between nodes are undirected and have a weight representing the distance between those two nodes.

### Problem dynamics

For simplicity, in this problem we assume that calls do not come in while ambulances are in transit. The agent will choose a location for each ambulance to move to, the ambulances move to that location, a call comes in, and the nearest ambulance travels to that location. There is no step for ambulances to travel to a hospital, and at the beginning of the next round, the ambulance that traveled to the call is still at that location.

## Environment

### Observation space

Each ambulance can be at any node in the graph, and multiple ambulances can be at the same node

### Action space

The agent chooses a node for each ambulance to travel to between calls

### Value function calculation:

The value function is $-1 * (\alpha * (distance traveled by ambulances from old state to action) + (1 - \alpha) * (distance traveled by ambulance from action to new arrival))$

The $\alpha$ parameter allows you to control the proportional difference in cost to move ambulances normally versus when responding to an emergency. This makes sense because you would likely want to penalize having to travel a long distance to reach a patient more heavily because there is a social, human cost in addition to a physical cost (i.e. gas).

Thought: should we penalize inequity in some way? penalty for having a high variance in response times?

By collecting data on their past actions, call arrival locations, and associated rewards, an agent's goal is to learn how to most effectively position ambulances to respond to calls.


`reset`

Returns the environment to its original state.

`step(action)`

Takes in the action from the agent and returns the state of the system of the next arrival.
* `action`: a list with the location of each ambulance

Ex. two ambulances at nodes 0 and 6 would be `[0, 6]`

Returns:

* `state`: A list containing the locations of each ambulance

* `reward`: The reward associated with the most recent action/event

* `pContinue`:

* `info`: a dictionary containing the node where the most recent arrival occured
  - Ex. `{'arrival': 1}` if the most recent arrival was at node 1

`render`

Currently unimplemented

`close`

Currently unimplemented

`make_ambulanceGraphEnvMDP(epLen, alpha, edges, starting_state, num_ambulance)`

Creates an instance of the environment

* `epLen`: is the length of each episode

* `alpha`: controls the proportional difference between the cost to move ambulances in between calls and the cost to move an ambulance to respond to a call.
  - `alpha = 0`: no cost to move between calls
  - `alpha = 1`: no cost to move to respond to a call

* `edges`: a list of tuples where each tuple has three entries corresponding to the starting node, the ending node, and the distance between them. The distance is a dictionary with one entry, 'dist', where the value is the distance
  - Ex. `(0, 4, {'dist': 2})` is an edge between nodes 0 and 4 with distance 2
  - The graph is undirected and nodes are inferred from the edges
  - Requires that the graph is fully connected
  - Requires that the numbering of nodes is chronological and starts at 0 (ie, if you have 5 nodes they must be labeled 0, 1, 2, 3, and 4)

* `starting_state`: a list where each index corresponds to an ambulance, and the entry at that index is the node where the ambulance is located

* `num_ambulance`: integer representing the number of ambulances in the system (kind of redundant, maybe we should get rid of this?)



**This will probably eventually go in a different file but for now I'm putting it here**

## Heuristic Agents

### Stable Agent

The stable agent chooses the location where each ambulance is already stationed as its action. Ambulances only move when responding to a call. 

### 'Median' Agent

The 'median' agent tries to combine choosing nodes where calls frequently occur with nodes that are relatively centrally located. The distance between each pair of nodes is calculated and put into a matrix, and the mean of each row is calculated. The entry for each node is then the mean distance between it and every other node in the graph. The inverse of this distance measurement for each node is multiplied by the number of times calls have come to that node in the past to calculate a score for each node, and ambulances will be sent to the top nodes.

### Mode Agent

The mode agent chooses the nodes where the most calls have come in the past.