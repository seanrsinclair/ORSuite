

`reset`

Returns the environment to it's original state.

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