#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 10:33:51 2021

@author: mayleencortez
"""
import numpy as np
from dynamics_model_4groups import dynamics_model

master_seed = 1

# population counts
n = 10000
n1 = 0.1 * n #1000
n2 = 0.2 * n #2000
n3 = 0.1 * n #1000
n4 = 0.6 * n #6000

# probabilities of becoming symptomatic if infected
p1 = 0.15
p2 = 0.15
p3 = 0.7
p4 = 0.2

# probabilities of being hospitalized if symptomatic
h1 = 0.2
h2 = 0.2
h3 = 0.7
h4 = 0.3

# contact matrix stuff
h = 2 / n
l = 0.5 / n
lambda_is = 3 / n
lambda_matrix = np.array([[h,h,l,l],[0,h,h,h],[0,0,l,l],[0,0,0,l]])

# vaccination and recovery rates
gamma = 0.006*n
beta = 1/7

priority_order = []
vaccines = n/4

# ignore for now, unimplemented in vaccine_model_v2
time_step = 0

starting_state = np.array([n1-10, n2-10, n3-10, n4-10, 10, 10, 10, 10, 0, 0, 0])
default_parameters = {'contact_matrix': lambda_matrix, 'lambda_hosp': lambda_is,
                      'rec': 0,
                      'p1': p1, 'p2': p2, 'p3': p3, 'p4': p4,
                      'h1': h1, 'h2': h2, 'h3': h3, 'h4': h4,
                      'gamma': gamma, 'beta': beta,
                      'priority_order': priority_order, 'vaccines': vaccines,
                      'time_step': time_step}

np.random.seed(master_seed)

newState, info = dynamics_model(parameters=default_parameters, population=starting_state)

#print(info.keys())
