"""
Adapted from code by Cornell University students Mohammad Kamil (mk848), Carrie Rucker (cmr284), Jacob Shusko (jws383), Kevin Van Vorst (kpv23)
"""
import numpy as np
import random
master_seed = 1

def dynamics_model(params, pop):
    """
    A function to run SIR disease dynamics for 4 groups.
    
    Args
    ----
    params: (dict) a dictionary containing the following keys and values
        'contact_matrix': (np.array of floats) contact rates between susceptible people in each class and the infected people
        'P': (np.array of floats) P = [p1 p2 p3 p4] where pi = Prob(symptomatic | infected) for a person in class i
        'H': (np.array of floats) H = [h1 h2 h3 h4] where hi = Prob(hospitalized | symptomatic) for a person in class i
        'beta': (float) recovery rate
        'gamma': (int) vaccination rate
        'vaccines': (int) number of vaccine available for this time period
        'priority': (list of chars) vaccination priority order of the four groups
        'time_step': (float) number of units of time you want the simulation to run for
            e.g. if all your rates are per day and you want to simulate 7 days, time_step = 7
    pop : (np.array of ints) the starting state [S1 S2 S3 S4 A1 A2 A3 A4 I H R]

    Returns
    -------
    newState : (np.array of ints) the final state [S1 S2 S3 S4 A1 A2 A3 A4 I H N]
        Note that instead of returning the final number of recovered people R, we return N, the number of infections that occurred
    output_dictionary : (dict) a dictionary containing the following keys and values
        'clock times': list of the times that each event happened
        'c1 asymptomatic': list of counts of A1 for each time in clks
        'c2 asymptomatic': list of counts of A2 for each time in clks
        'c3 asymptomatic': list of counts of A3 for each time in clks
        'c4 asymptomatic': list of counts of A4 for each time in clks
        'mild symptomatic': list of counts of I for each time in clks
        'hospitalized': list of counts of H for each time in clks
        'c1 susceptible': list of counts of S1 for each time in clks
        'c2 susceptible': list of counts of S2 for each time in clks
        'c3 susceptible': list of counts of S3 for each time in clks
        'c4 susceptible': list of counts of S4 for each time in clks
        'recovered': list of counts of R for each time in clks
        'total infected': int - total number of infected (including those that were already infected)
        'total hospitalized': int - total number of hospitalized individuals (including those that were already hospitalized)
        'vaccines': int - total number of vaccines left
        
            
        
    Typical usage example
    ----------------------
    newState, info = dynamics_model(parameters, population)
    """

    # extract arguments from params dictionary
    state = pop
    P = params['P']
    H = params['H']
    LAMBDA = params['contact_matrix']
    gamma = params['gamma']
    beta = params['beta']
    priority = params['priority_order']
    vaccines = params['vaccines']
    time_step = params['time_step']

    # output tracking
    clks = [0]
    c1_Ss = [state[0]]
    c2_Ss = [state[1]]
    c3_Ss = [state[2]]
    c4_Ss = [state[3]]
    c1_infs = [state[4]]
    c2_infs = [state[5]]
    c3_infs = [state[6]]
    c4_infs = [state[7]] 
    Is_infs = [state[8]]
    Hs_infs = [state[9]]
    Rs = [state[10]]
    
    # first priority group
    if len(priority != 0):
        priority_group = int(priority[0]) - 1
        priority.pop(0)
        randomFlag = False
    else:
        eligible = [0,1,2,3]
        priority_group = random.choice(eligible)
        randomFlag = True
    
    # possible state changes
    # each key correponds to the index of an event in rates and has a value [i,j]
    # the state change is state[i]-- and state[j]++
    state_changes = {0: [0,4], 1: [0,8], 2: [0,9], 3: [1,5],
                     4: [1,8], 5: [1,9], 6: [2,6], 7: [2,8],
                     8: [2,9], 9: [3,7], 10: [3,8], 11: [3,9],
                     12: [4,10], 13: [5,10], 14: [6,10], 15: [7,10],
                     16: [8,10], 17: [9,10], 18: [0,10], 19: [1,10],
                     20: [2,10], 21: [3,10]}

    # rates for all 22 events
    rates = np.zeros(shape=(1,22))
    
    # counts for each of the 22 events
    event_counts = np.zeros(shape=(1,22))
    
    # compute the probabilities associated with each of the 12 infection rates
    probs = np.zeros(shape=(1,12))
    probs[[0,3,6,9]] = 1 - P
    probs[[1,4,7,10]] = np.multiply(P,1-H)
    probs[[2,5,8,11]] = np.multiply(P,H)
    
    # compute the rates for the 12 infection events
    inf_rates = np.matmul(np.matmul(np.diag(state[0:3]),LAMBDA),state[4:9])
    inf_rates = np.repeat(inf_rates, repeats = 3, axis = 0)
    rates[0:11] = np.multiply(probs, inf_rates)
    
    # compute the rates for the 6 recovery events
    rates[12:17] = beta * state[4:9]
    
    # compute the rates for the vaccination events
    rates[priority_group + 18] = gamma
    
    # flag - if true, we have not run out of vaccines or people to vaccinate yet 
    #      - if false, either there are no vaccines left or no people to vaccinate
    # once set to False, it remains False
    vaccFlag = True

    rate_sum = np.sum(rates)

    # exponential timer
    nxt = np.random.exponential(1/rate_sum)
    clk = 0
    
    # maximum number of vaccination events that we want to happen
    max_vacc_events = gamma*time_step

    # We will simulate the Markov chain until we've reached max_vacc_events vaccination events
    while np.sum(event_counts[18:21]) < max_vacc_events: 
        clk += nxt
        
        # get the index of the event that is happening
        index = np.random.choice(22, 1, p = rates/rate_sum)
        
        # if this is a vaccination event, call vacc_update
        # otherwise, simple state change
        if index in np.arange(18,22):
            if randomFlag:
                state, event_counts, priority_group, eligible, vaccines = rand_vacc_update(state=state, 
                                                                            changes=state_changes, 
                                                                            group=priority_group, 
                                                                            eligible=eligible, 
                                                                            vaccines=vaccines, 
                                                                            count=event_counts)
                    
            else:
                state, event_counts, vaccFlag, priority_group, priority, vaccines = vacc_update(state=state, 
                                                                                            changes=state_changes, 
                                                                                            ind=index, 
                                                                                            count=event_counts, 
                                                                                            flag=vaccFlag, 
                                                                                            group=priority_group, 
                                                                                            priority=priority, 
                                                                                            vaccines=vaccines)
            # update vaccination rate
            rates[18:21] = np.zeros(shape=(1,4))
            rates[priority_group+18] = gamma
        else:
            state[state_changes[index][0]] -= 1
            state[state_changes[index][1]] += 1
            event_counts[index] += 1

        # update infection and recovery rates 
        ## 12 infection events
        inf_rates = np.matmul(np.matmul(np.diag(state[0:3]),LAMBDA),state[4:9])
        inf_rates = np.repeat(inf_rates, repeats = 3, axis = 0)
        rates[0:11] = np.multiply(probs, inf_rates)
        
        ## 6 recovery events
        rates[12:17] = beta * state[4:9]
    
        rate_sum = np.sum(rates)
    
        # TODO: not sure if this conditional is necessary
        if rate_sum > 0:
            nxt = np.random.exponential(1/rate_sum)
        else:
            print("The sum of the rates is less than or equal to zero!")
            break
 
        # output tracking
        clks.append(clk)
        c1_Ss.append(state[0])
        c2_Ss.append(state[1])
        c3_Ss.append(state[2])
        c4_Ss.append(state[3])
        c1_infs.append(state[4])
        c2_infs.append(state[5])
        c3_infs.append(state[6])
        c4_infs.append(state[7])
        Is_infs.append(state[8])
        Hs_infs.append(state[9])
        Rs.append(state[10])

        # if there are no more infected individuals, the simulation should end
        if np.sum(state[4:10]) <= 0:
            print("Reached a disease-free state on day " + str(clk))

    new_infections = np.sum(event_counts[0:11])
    total_infected = new_infections + np.sum(pop[4:9])
    total_hospitalized = np.sum(event_counts[2,5,8,11]) + pop[9]
    newState = state

    output_dictionary = {'clock times': clks, 'c1 asymptomatic': c1_infs, 'c2 asymptomatic': c2_infs, 'c3 asymptomatic': c3_infs,
                         'c4 asymptomatic': c4_infs, 'mild symptomatic': Is_infs, 'hospitalized': Hs_infs, 'c1 susceptible': c1_Ss,
                         'c2 susceptible': c2_Ss, 'c3 susceptible': c3_Ss, 'c4 susceptible': c4_Ss, 'recovered': Rs, 'total infected': total_infected,
                         'total hospitalized': total_hospitalized, 'vaccines': vaccines}
    return newState, output_dictionary

def vacc_update(state, changes, ind, count, flag, group, priority, vaccines):
    newState = state
    if flag:
        if vaccines > 0:
            newState[changes[ind][0]] -= 1
            newState[changes[ind][1]] += 1
            vaccines -= 1
            while np.any(newState < 0):
                newState = state
                vaccines += 1
                if len(priority) != 0:
                    group = int(priority[0]) - 1
                    priority.pop(0)
                    newState[changes[group+18][0]] -= 1
                    newState[changes[group+18][1]] += 1
                    vaccines -= 1
                else:
                    flag = False
            count[group+18] += 1
                    
        else:
            count[ind] += 1
            flag = False
    else:
        count[ind] += 1
    return newState, count, flag, group, priority, vaccines
    
def rand_vacc_update(state, changes, group, eligible, vaccines, count):
    if len(eligible) != 0:
        state[changes[group+18][0]] -= 1
        state[changes[group+18][1]] += 1
        vaccines -= 1
        while np.any(state < 0):
            state[changes[group+18][0]] += 1
            state[changes[group+18][1]] -= 1
            vaccines += 1
            if len(eligible) != 0:
                eligible.remove(group)
                group = random.choice(eligible)
                state[changes[group+18][0]] -= 1
                state[changes[group+18][1]] += 1
                vaccines -= 1
            else:
                break
        count[group+18] += 1
    else:
        count[group+18] += 1
    return state, count, group, eligible, vaccines