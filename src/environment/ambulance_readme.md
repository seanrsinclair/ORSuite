## Overview

### Description

`ambulance.py` is a reinforcement learning environment meant to represent a situation where ambulances can be stationed at different locations, and the nearest one will respond to an incoming call, and the locations are all located along a single line. Ambulances can also move to new locations between calls. The environment is represented by a number line between 0 and 1.

### Problem dynamics

For simplicity, in this problem we assume that calls do not come in while ambulances are in transit. The agent will choose a location for each ambulance to move to, the ambulances move to that location, a call comes in, and the nearest ambulance travels to that location. There is no step for ambulances to travel to a hospital, and at the beginning of the next round, the ambulance that traveled to the call is still at that location.

## Environment

### Observation space

Each ambulance can be anywhere on the line

### Action space

The agent chooses a location for each ambulance to travel to between calls

### Value function calculation

The value function is $-1 * (\alpha * (distance traveled by ambulances from old state to action) + (1 - \alpha) * (distance traveled by ambulance from action to new arrival))$

The $\alpha$ parameter allows you to control the proportional difference in cost to move ambulances normally versus when responding to an emergency. This makes sense because you would likely want to penalize having to travel a long distance to reach a patient more heavily because there is a social, human cost in addition to a physical cost (i.e. gas).

By collecting data on their past actions, call arrival locations, and associated rewards, an agent's goal is to learn how to most effectively position ambulances to respond to calls.

`reset`

`step`

`render`

`close`

`make_ambulanceEnvMDP()`



**This will probably eventually go in a different file but for now I'm putting it here**

## Heuristic Agents

### Stable Agent

The stable agent chooses the location where each ambulance is already stationed as its action. Ambulances only move when responding to a call. 

### K-Medoid Agent

The k-medoid agent uses the k-medoid algorithm where k is the number of ambulances to figure out where to station ambulances.