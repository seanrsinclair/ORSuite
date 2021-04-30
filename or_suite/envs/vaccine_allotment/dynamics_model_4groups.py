"""
Adapted from code by Cornell University students Mohammad Kamil (mk848), Carrie Rucker (cmr284), Jacob Shusko (jws383), Kevin Van Vorst (kpv23)
"""
import numpy as np
master_seed = 1

def dynamics_model(parameters, population):
    """
    A function to run SIR disease dynamics for 4 groups.
    
    Args
    ----
    parameters : dict
        DESCRIPTION.
    population : numpy array with 11 entries
        DESCRIPTION.

    Returns
    -------
    newState : numpy array wtih 11 entries
        DESCRIPTION.
    output_dictionary : dict
        DESCRIPTION.
        
    Typical usage example
    ----------------------
    newState, info = dynamics_model(parameters, population)
    """
    
    # extract population sizes from population np.array
    c1_s, c2_s, c3_s, c4_s = population[0], population[1], population[2], population[3] # susceptible people per class 
    c1_ia, c2_ia, c3_ia, c4_ia = population[4], population[5], population[6], population[7] # asymptomatic people per class
    Is = population[8] # aggregate count for mild symptomatic infections
    Hs = population[9] # aggregate count for hospitalized infected people for all classes
    R = 0 # aggregate count for recovered people for all classes

    # extract parameters from params dictionary
    p1, p2, p3, p4 = parameters['p1'], parameters['p2'], parameters['p3'], parameters['p4']
    h1, h2, h3, h4 = parameters['h1'], parameters['h2'], parameters['h3'], parameters['h4']
    lambda_matrix = parameters['contact_matrix']
    lambda_is = parameters['lambda_hosp']
    gamma = parameters['gamma']
    beta = parameters['beta']
    priority_order = parameters['priority_order']
    vaccines = parameters['vaccines']
    time_step = parameters['time_step']

    # output tracking
    clks = [0]
    c1_infs = [c1_ia]
    c2_infs = [c2_ia]
    c3_infs = [c3_ia]
    c4_infs = [c4_ia] 
    Is_infs = [Is]
    Hs_infs = [Hs]
    c1_Ss = [c1_s]
    c2_Ss = [c2_s]
    c3_Ss = [c3_s]
    c4_Ss = [c4_s]
    Rs = [R]
    total_infected = c1_ia + c2_ia + c3_ia + c4_ia + Is + Hs
    total_hospitalized = Hs
    # failed_vaccines = 0
    hosp_flag = False
    new_infections = total_infected 

    # rates
    c1_c1_rate = lambda_matrix[0,0]*2*c1_s*c1_ia
    c2_c2_rate = lambda_matrix[1,1]*2*c2_s*c2_ia
    c3_c3_rate = lambda_matrix[2,2]*2*c3_s*c3_ia
    c4_c4_rate = lambda_matrix[3,3]*2*c4_s*c4_ia

    c1_c2_rate = lambda_matrix[0,1]*(c1_s*c2_ia + c2_s*c1_ia)
    c1_c3_rate = lambda_matrix[0,2]*(c1_s*c3_ia + c3_s*c1_ia)
    c1_c4_rate = lambda_matrix[0,3]*(c1_s*c4_ia + c4_s*c1_ia)

    c2_c3_rate = lambda_matrix[1,2]*(c2_s*c3_ia + c3_s*c2_ia)
    c2_c4_rate = lambda_matrix[1,3]*(c2_s*c4_ia + c4_s*c2_ia)

    c3_c4_rate = lambda_matrix[2,3]*(c3_s*c4_ia + c4_s*c3_ia)

    c1_is_rate = lambda_is*c1_s*Hs

    healing_rate = beta*(c1_ia+c2_ia+c3_ia+c4_ia+Hs+Is)
    vaccine_rate = gamma

    rate_sum = (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate + 
              c1_c4_rate + c2_c3_rate + c2_c4_rate + c3_c4_rate + c1_is_rate + healing_rate + vaccine_rate)

    nxt = np.random.exponential(1/rate_sum)
    clk = 0

    if time_step <= 0:
        end_condition = lambda: (c1_ia > 0 or c2_ia > 0 or c3_ia > 0 or c4_ia > 0 or Hs > 0 or Is > 0)
    else:
        end_condition = lambda: (clk <= time_step)

    while end_condition(): 
        clk += nxt
        prob = np.random.uniform()

        ## intraclass meetings
        # c1 <-> c1
        if prob < c1_c1_rate / rate_sum:
            c1_ia, c1_s, Hs, total_infected, Is, hosp_flag = intraclass_meeting(ia=c1_ia,s=c1_s,p=p1,h=h1,Hs=Hs,infected=total_infected,Is=Is)
 
        # c2 <-> c2
        elif prob < (c1_c1_rate + c2_c2_rate)  / rate_sum:
            c2_ia, c2_s, Hs, total_infected, Is, hosp_flag = intraclass_meeting(ia=c2_ia,s=c2_s,p=p2,h=h2,Hs=Hs,infected=total_infected,Is=Is)
 
        # c3 <-> c3
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate)  / rate_sum:
            c3_ia, c3_s, Hs, total_infected, Is, hosp_flag = intraclass_meeting(ia=c3_ia,s=c3_s,p=p3,h=h3,Hs=Hs,infected=total_infected,Is=Is)
    
        # c4 <-> c4
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate)  / rate_sum:
            c4_ia, c4_s, Hs, total_infected, Is, hosp_flag = intraclass_meeting(ia=c4_ia,s=c4_s,p=p4,h=h4,Hs=Hs,infected=total_infected,Is=Is)
 
        ## interclass meetings
        # c1 <-> c2
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate)  / rate_sum:
            c1_ia, c1_s, c2_ia, c2_s, Hs, total_infected, Is, hosp_flag = interclass_meeting(ia_1st=c1_ia,s_1st=c1_s,ia_2nd=c2_ia,s_2nd=c2_s,p_1st=p1,p_2nd=p2,h_1st=h1,h_2nd=h2,Hs=Hs,infected=total_infected,Is=Is)
 
        # c1 <-> c3
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate)  / rate_sum:
            c1_ia, c1_s, c3_ia, c3_s, Hs, total_infected, Is, hosp_flag = interclass_meeting(ia_1st=c1_ia,s_1st=c1_s,ia_2nd=c3_ia,s_2nd=c3_s,p_1st=p1,p_2nd=p3,h_1st=h1,h_2nd=h3,Hs=Hs,infected=total_infected,Is=Is)
    
        # c1 <-> c4
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate + c1_c4_rate)  / rate_sum:
            c1_ia, c1_s, c4_ia, c4_s, Hs, total_infected, Is, hosp_flag = interclass_meeting(ia_1st=c1_ia,s_1st=c1_s,ia_2nd=c4_ia,s_2nd=c4_s,p_1st=p1,p_2nd=p4,h_1st=h1,h_2nd=h4,Hs=Hs,infected=total_infected,Is=Is)
    
        # c2 <-> c3
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate + c1_c4_rate + c2_c3_rate)  / rate_sum:
            c2_ia, c2_s, c3_ia, c3_s, Hs, total_infected, Is, hosp_flag = interclass_meeting(ia_1st=c2_ia,s_1st=c2_s,ia_2nd=c3_ia,s_2nd=c3_s,p_1st=p2,p_2nd=p3,h_1st=h2,h_2nd=h3,Hs=Hs,infected=total_infected,Is=Is)
 
        # c2 <-> c4
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate + c1_c4_rate + c2_c3_rate + c2_c4_rate)  / rate_sum:
            c2_ia, c2_s, c4_ia, c4_s, Hs, total_infected, Is, hosp_flag = interclass_meeting(ia_1st=c2_ia,s_1st=c2_s,ia_2nd=c4_ia,s_2nd=c4_s,p_1st=p2,p_2nd=p4,h_1st=h2,h_2nd=h4,Hs=Hs,infected=total_infected,Is=Is)
 
        # c3 <-> c4
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate + c1_c4_rate + c2_c3_rate + c2_c4_rate + c3_c4_rate)  / rate_sum:
            c3_ia, c3_s, c4_ia, c4_s, Hs, total_infected, Is, hosp_flag = interclass_meeting(ia_1st=c3_ia,s_1st=c3_s,ia_2nd=c4_ia,s_2nd=c4_s,p_1st=p3,p_2nd=p4,h_1st=h3,h_2nd=h4,Hs=Hs,infected=total_infected,Is=Is)
 
        # c1 <-> Hs
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate + c1_c4_rate + c2_c3_rate + c2_c4_rate + c3_c4_rate + c1_is_rate)  / rate_sum:
            U = np.random.uniform()
            if c1_s > 0:
                c1_s -= 1
                total_infected += 1
                if U < p1:
                    U2 = np.random.uniform()
                    if U2 < h1:
                        Hs += 1
                        hosp_flag = True
                    else:
                        Is += 1
                else:
                    c1_ia += 1
 
        # healing
        elif prob < (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate + c1_c4_rate + c2_c3_rate + c2_c4_rate + c3_c4_rate + c1_is_rate + healing_rate)  / rate_sum:
            U = np.random.uniform()
            inf_sum = c1_ia + c2_ia + c3_ia + c4_ia + Hs + Is
            if inf_sum > 0:
                R += 1
                if U < c1_ia / inf_sum:
                    c1_ia -= 1
                elif U < (c1_ia + c2_ia) / inf_sum:
                    c2_ia -= 1
                elif U < (c1_ia + c2_ia + c3_ia) / inf_sum:
                    c3_ia -= 1
                elif U < (c1_ia + c2_ia + c3_ia + c4_ia) / inf_sum:
                    c4_ia -= 1
                elif U < (c1_ia + c2_ia + c3_ia + c4_ia + Hs) / inf_sum:
                    Hs -= 1
                else:
                    Is -= 1
 
        # vaccination
        else:
            susc_sum = c1_s + c2_s + c3_s + c4_s
            if (vaccines > 0 and susc_sum > 0):
                R += 1
                vaccines -= 1
                if priority_order == []:
                    U = np.random.uniform()
                    if susc_sum > 0:
                        if U < c1_s / susc_sum:
                            c1_s -= 1
                        elif U < (c1_s + c2_s) / susc_sum:
                            c2_s -= 1
                        elif U < (c1_s + c2_s + c3_s) / susc_sum:
                            c3_s -= 1
                        else:
                            c4_s -= 1
                else:
                    if priority_order[0] == "c1":
                        if c1_s > 0:
                            c1_s -= 1
                        else:
                            priority_order.pop(0)
                    elif priority_order[0] == "c2":
                        if c2_s > 0:
                            c2_s -= 1
                        else:
                            priority_order.pop(0)
                    elif priority_order[0] == "c3":
                        if c3_s > 0:
                            c3_s -= 1
                        else:
                            priority_order.pop(0)
                    else:
                        if c4_s > 0:
                            c4_s -= 1
                        else:
                            priority_order.pop(0)
         
        # update variable rates and calculate next event [nxt] 
        c1_c1_rate = lambda_matrix[0,0]*2*c1_s*c1_ia
        c2_c2_rate = lambda_matrix[1,1]*2*c2_s*c2_ia
        c3_c3_rate = lambda_matrix[2,2]*2*c3_s*c3_ia
        c4_c4_rate = lambda_matrix[3,3]*2*c4_s*c4_ia
     
        c1_c2_rate = lambda_matrix[0,1]*(c1_s*c2_ia + c2_s*c1_ia)
        c1_c3_rate = lambda_matrix[0,2]*(c1_s*c3_ia + c3_s*c1_ia)
        c1_c4_rate = lambda_matrix[0,3]*(c1_s*c4_ia + c4_s*c1_ia)
     
        c2_c3_rate = lambda_matrix[1,2]*(c2_s*c3_ia + c3_s*c2_ia)
        c2_c4_rate = lambda_matrix[1,3]*(c2_s*c4_ia + c4_s*c2_ia)
     
        c3_c4_rate = lambda_matrix[2,3]*(c3_s*c4_ia + c4_s*c3_ia)
     
        c1_is_rate = lambda_is*c1_s*Hs
     
        healing_rate = beta*(c1_ia+c2_ia+c3_ia+c4_ia+Hs+Is)
    
        rate_sum = (c1_c1_rate + c2_c2_rate + c3_c3_rate + c4_c4_rate + c1_c2_rate + c1_c3_rate + c1_c4_rate + c2_c3_rate + c2_c4_rate + c3_c4_rate + c1_is_rate + healing_rate + vaccine_rate)
    
        if rate_sum > 0:
            nxt = np.random.exponential(1/rate_sum)
        else:
            break
 
        # output tracking
        clks.append(clk)
        c1_infs.append(c1_ia)
        c2_infs.append(c2_ia)
        c3_infs.append(c3_ia)
        c4_infs.append(c4_ia) 
        Is_infs.append(Is)
        Hs_infs.append(Hs)
        c1_Ss.append(c1_s)
        c2_Ss.append(c2_s)
        c3_Ss.append(c3_s)
        c4_Ss.append(c4_s)
        Rs.append(R)
        if hosp_flag:
            total_hospitalized += 1
            hosp_flag = False

        if c1_ia + c2_ia + c3_ia + c4_ia + Hs + Is <= 0:
            #print("Reached a disease-free state on day " + str(clk))
            break

    new_infections = total_infected - new_infections

    newState = np.array([c1_s, c2_s, c3_s, c4_s, c1_ia, c2_ia, c3_ia, c4_ia, Is, Hs, new_infections])

    output_dictionary = {'clock times': clks, 'c1 asymptomatic': c1_infs, 'c2 asymptomatic': c2_infs, 'c3 asymptomatic': c3_infs,
                         'c4 asymptomatic': c4_infs, 'mild symptomatic array': Is_infs, 'hospitalized array': Hs_infs, 'c1 susceptible': c1_Ss,
                         'c2 susceptible': c2_Ss, 'c3 susceptible': c3_Ss, 'c4 susceptible': c4_Ss, 'recovered': Rs, 'total infected': total_infected,
                         'total hospitalized': total_hospitalized, 'vaccines': vaccines}
    return newState, output_dictionary


def intraclass_meeting(ia,s,p,h,Hs,infected,Is,flag=False):
    """
    A function to represent an interaction within a group (i.e. group i with group i).
    
    This function is called by dynamics_model and shouldn't be called externally by the user.
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
    
    This function is called by dynamics_model and shouldn't be called externally by the user.
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
