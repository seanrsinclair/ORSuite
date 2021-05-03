"""
Adapted from code by Cornell University students Mohammad Kamil (mk848), Carrie Rucker (cmr284), Jacob Shusko (jws383), Kevin Van Vorst (kpv23)
"""
import numpy as np
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
    '''
    # extract population sizes from population np.array
    c1_s, c2_s, c3_s, c4_s = pop[0], pop[1], pop[2], pop[3] # susceptible people per class 
    c1_ia, c2_ia, c3_ia, c4_ia = pop[4], pop[5], pop[6], pop[7] # asymptomatic people per class
    Is = pop[8] # aggregate count for mild symptomatic infections
    Hs = pop[9] # aggregate count for hospitalized infected people for all classes
    R = 0 # aggregate count for recovered people for all classes
    '''

    # extract parameters from params dictionary
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
    total_infected = np.sum(state[4:9])
    total_hospitalized = state[9]
    # failed_vaccines = 0
    hosp_flag = False
    new_infections = 0

    # initialize variable that will hold all 22 event rates
    rates = np.zeros(shape=(1,22))
    
    # compute the rates for the 12 infection events
    

    rate_sum = np.sum(rates)

    nxt = np.random.exponential(1/rate_sum)
    clk = 0

    if time_step <= 0:
        end_condition = lambda: (np.sum(state[4:9]) > 0)
    else:
        end_condition = lambda: (clk <= time_step)

    while end_condition(): 
        clk += nxt
        prob = np.random.uniform()
        pass

        # update variable rates and calculate next event [nxt] 
        rates[0:3] = np.multiply(np.matmul(LAMBDA,state[4:10]),state[0:3])
        rates[4] = beta*np.sum(state[4:10]) # recovery rate
        rates[5] = gamma # vaccination rate
    
        rate_sum = np.sum(rates)
    
        if rate_sum > 0:
            nxt = np.random.exponential(1/rate_sum)
        else:
            break
 
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
        if hosp_flag:
            total_hospitalized += 1
            hosp_flag = False

        if np.sum(state[4:10]) <= 0:
            #print("Reached a disease-free state on day " + str(clk))
            break

    new_infections = total_infected - new_infections

    newState = state

    output_dictionary = {'clock times': clks, 'c1 asymptomatic': c1_infs, 'c2 asymptomatic': c2_infs, 'c3 asymptomatic': c3_infs,
                         'c4 asymptomatic': c4_infs, 'mild symptomatic': Is_infs, 'hospitalized': Hs_infs, 'c1 susceptible': c1_Ss,
                         'c2 susceptible': c2_Ss, 'c3 susceptible': c3_Ss, 'c4 susceptible': c4_Ss, 'recovered': Rs, 'total infected': total_infected,
                         'total hospitalized': total_hospitalized, 'vaccines': vaccines}
    return newState, output_dictionary


def intraclass_meeting(ia,s,p,h,Hs,infected,Is,flag=False):
    """
    A function to represent an interaction within a group (i.e. group i with group i).
    
    Note: This function is called by dynamics_model and shouldn't be called externally by the user.
    
    Args
    ----
    
    """
    U = np.random.uniform()
    infs = infected
    infs_mild = Is
    if s > 0:
        s -= 1
        infs += 1
        if U < p:
            U2 = np.random.uniform()
            if U2 < h:
                Hs += 1
                flag = True
            else:
                infs_mild += 1
        else:
            ia += 1
    return ia,s,Hs,infs,infs_mild, flag


def interclass_meeting(ia_1st,s_1st,ia_2nd,s_2nd,p_1st,p_2nd,h_1st,h_2nd,Hs,infected,Is,flag=False):
    """
    A function implementing an interaction between two different groups (i.e. group i with group j where i =/= j).
    
    Note: This function is called by dynamics_model and shouldn't be called externally by the user.
    
    Args
    ----
    
    """
    prop = (ia_1st + s_2nd) / (s_1st + ia_1st + s_2nd + ia_2nd)
    U1 = np.random.uniform()
    U2 = np.random.uniform()
    infs = infected
    infs_mild = Is
    if U1 < prop:
        if s_2nd > 0:
            s_2nd -= 1
            infs += 1
            if U2 < p_2nd:
                U3 = np.random.uniform()
                if U3 < h_2nd:
                    Hs += 1
                    flag = True
                else:
                    infs_mild += 1
            else:
                ia_2nd += 1
    else:
        if s_1st > 0:
            s_1st -= 1
            infs += 1
            if U2 < p_1st:
                U3 = np.random.uniform()
                if U3 < h_1st:
                    Hs += 1
                    flag = True
                else:
                    infs_mild += 1
            else:
                ia_1st += 1
    return ia_1st,s_1st,ia_2nd,s_2nd,Hs,infs,infs_mild,flag
