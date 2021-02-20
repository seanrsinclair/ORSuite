Markdown
### Code Overview
​
`reset`
​
Returns the environment to it's original state.
​
`step(action)`
​

Takes in the action from the agent and returns the state of the system of the next arrival.

* `action`: A $K$-dimensional vector denoting the allocation: how much of each resource is given


Returns:
​
* `state`: A tuple containing, the budget remaining, the type vector and the relative size of the next location

* `reward`: The reward associated with the most recent action/event. Currently is the log of the utility function

* `pContinue`: information on whether or not the episode should continue
​
* `info`: a dictionary containing the type of the newest location
  - Ex. `{'type': np.array([1,2,3])}` if the newest location has type of [1,2,3]

`render`

Currently unimplemented

`close`

Currently unimplemented

`make_resource_allocationEnvMDP(K, num_agents, weight_matrix, init_budget, endowments, type_dist, u)`

Creates an instance of the environment

* `K`: number of commodities available

* `num_agents`: number of locations to visit

* `weight_matrix`: An $n \times K$ matrix where each row denotes a possible type (there are $n$ possible types)

* `init_budget`: $K$-dimensional vector indicating the initial budget for all commodities

* `endowments`: a vector detailing the relative size of each location 

* `type_dist`: a function mapping integers to distributions over $\{1,...,n\}$, details the types each location might have
    - Ex. `type_dist(0) = np.random.randint(0,n)` means location 0 is equally likely to have any one of the $n$ types

* `u`: the utility function, takes in an allocation vector and a type vector and returns a real number denoting the utility
    - Ex. `u(x,theta) = np.dot(x,theta)` means the utility from giving an allocation $x$ to a location of type $\theta$ is $\langle x , \theta \rangle$ 

* `starting_state`: a tuple of `(init_budget, type_dist(0))`

### Formal Definition 

* We consider the MDP formed by the sequential allocation problem a 5-tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, R, \mathcal{H})$
    - Our state space $\mathcal{S} := \{(b,\theta_i)|b \in \mathbb{R}_+^{k},\theta_i \in \mathbb{R}_+^k\}$ where $b$ is a vector of all remaining budget for commodity $k$, and $\theta_i$ is a type vector indicating the preferences for each commodity for agent $i$. Our initial state $S_0 = (B,\theta_0)$, where $B$ is the full pre-planned budget and $\theta_0 \sim \mathcal{F}_0$ 
    - Actions correspond to the allocation we make to agent $i$. Formally action-space in state $i$ is defined as $A_i := \{X \in \mathbb{R}_+^k|X \leq b\}$ where $b$ is the current budget vector. 
    - Our reward-space $R$ is the Nash Social Welfare: while in state $s$ and taking action $a$, we have $R(s,a) = \log u(a,\theta)$ where $u: \mathbb{R}^k \times \mathbb{R}^k \to \mathbb{R}_+$ is a utility function for the agents.
    - Transitions. Given state $(b,\theta_i) \in \mathcal{S}$ and action $X \in \mathcal{A}$. we have our new state $s_{i+1} = (b-X,\theta_{i+1})$ where $\theta_{i+1} \sim \mathcal{F}_{i+1}$
    - Each episode will have the same number of steps as there are agents. Thus $\mathcal{H}=n$
​
