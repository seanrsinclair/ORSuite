import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib
import pickle
import numpy as np
import pandas as pd
from math import pi


def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        if feature_name != 'group':
                max_value = df[feature_name].max()
                min_value = df[feature_name].min()
                print(max_value, min_value, feature_name)
                result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
                
    return result


def plot_radar_plots(path1, path2 , fig_path , fig_name):
    
    nEps = 50
    numIters = 20
    epLen = 5

    #Organizing Data

    #Reward : Final episodes avg. over iters
    reward = pd.read_csv(path1)
    r2 = pd.read_csv(path2)
    reward = reward.loc[reward['episode'] == (nEps-1)]
    reward = reward['epReward'].mean()
    r2 = r2.loc[r2['episode'] == (nEps-1)]
    r2 = r2['epReward'].mean()
    reward = np.array([reward,r2])
    
    #Time : avg over iters
    time = pd.read_csv(path1).groupby('iteration').mean()
    t1 = pd.read_csv(path2).groupby('iteration').mean()
    time = time['time'].mean()
    t1 = t1['time'].mean()
    time = np.array([time , t1])
    
    #Space : avg over iters
    space = pd.read_csv(path1).groupby('iteration').mean()
    s1 = pd.read_csv(path2).groupby('iteration').mean()
    space = space['memory'].mean()
    s1 = s1['memory'].mean()
    space = np.array([space , s1])    

    # Set data
    df = pd.DataFrame({'group': ['Median', 'Stable'],
                       'Time': time,
                       'Space': space,
                       'Reward': np.abs(reward)})
    print(df)
    df = normalize(df)
    
    print(df)
    # ------- PART 1: Create background

    # number of variable
    categories=list(df)[1:]
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.style.use('seaborn')
    plt.xticks(angles[:-1], categories)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([.2,.4,.6,.8], [".2",".4",".6", ".8"], color="grey", size=7)
    plt.ylim(0,1.1)


    # ------- PART 2: Add plots

    # Plot each individual = each line of the data
    # I don't make a loop, because plotting more than 3 groups makes the chart unreadable

    # Ind1
    values=df.loc[0].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Median")
    ax.fill(angles, values, 'b', alpha=0.1)

    # Ind2
    values=df.loc[1].drop('group').values.flatten().tolist()
    values += values[:1]
    ax.plot(angles, values, linewidth=1, linestyle='solid', label="Stable")
    ax.fill(angles, values, 'r', alpha=0.1)

    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon = True)

    if os.path.exists(fig_path):
                plt.savefig(os.path.join(fig_path,fig_name), bbox_inches = 'tight',
                pad_inches = 0.01)
    else:
                os.makedirs(fig_path)
                plt.savefig(os.path.join(fig_path,fig_name), bbox_inches = 'tight',
                pad_inches = 0.01)
        

def plot_line_plots(path1, path2 , path3, fig_path , fig_name):
    

        dt_first = pd.read_csv(path_first).groupby(['episode']).mean()
        dt_second = pd.read_csv(path_sec).groupby(['episode']).mean()
        dt_third = pd.read_csv(path_third).groupby(['episode']).mean()
        dt_first['episode'] = dt_first.index.values
        dt_second['episode'] = dt_second.index.values
        dt_third['episode'] = dt_third.index.values
        
        #Selecting the number of points to plot default = 5
        dt_second = dt_second.iloc[::5, :]
        dt_first = dt_first.iloc[::5, :]
        dt_third = dt_third.iloc[::5, :]
       
        fig, ax = plt.subplots(1, 3, constrained_layout=False, figsize=(15,5))
        fig.suptitle('Comparison of Observed Rewards')
        
        ax[0].plot(dt_first['episode'], dt_first['epReward'], label='Random', linestyle=':')
        ax[0].plot(dt_second['episode'], dt_second['epReward'], label = 'Stable', linestyle='-.')
        ax[0].plot(dt_third['episode'], dt_third['epReward'], label = 'Median', linestyle='-')
        ax[0].set_ylabel('Observed Reward')


        ax[1].plot(dt_first['episode'], dt_first['time'], label='Random', linestyle=':')
        ax[1].plot(dt_second['episode'], dt_second['time'], label = 'Stable', linestyle='-.')
        ax[1].plot(dt_third['episode'], dt_third['time'], label='Median', linestyle='-')
        ax[1].set_ylabel('Observed Time Used ( ns )')

        ax[2].plot(dt_first['episode'], dt_first['memory'], label='Random', linestyle=':')
        ax[2].plot(dt_second['episode'], dt_second['memory'], label = 'Stable', linestyle='-.')
        ax[2].plot(dt_third['episode'], dt_third['memory'], label='Median', linestyle='-')
        ax[2].set_ylabel('Observed Memory Usage (B)')

        # plt.ylim(0,5+.1)
        plt.xlabel('Episode')
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        if os.path.exists(fig_path):
                fig.savefig(os.path.join(fig_path,fig_name), bbox_inches = 'tight',
                pad_inches = 0.01, dpi=900)
        else:
                os.makedirs(fig_path)
                fig.savefig(os.path.join(fig_path,fig_name), bbox_inches = 'tight',
                pad_inches = 0.01, dpi=900)
        
        
if __name__ == "__main__":
    
    nEps = 50
    numIters = 20
    epLen = 5

    list_algo_1 = []
    list_algo_2 = []

    env_type = 'ambulance_metric'
    problem_list = ['shifting', 'beta', 'uniform']
    param_list = ['0', '1', '0.25']
    algo_list = ['Random' , 'Stable', 'Median']

    my_path = os.path.abspath(__file__) # Figures out the absolute path for you in case your working directory moves around.
    
    for problem in problem_list:
        for param in param_list:

            path_first = '../data/'+env_type+'_'+str(algo_list[0])+'_'+param+'_'+problem+'/data.csv'
    #         name_median = '../data/ambulance'+problem+'_'+param+'_Median'+'/data.csv'
    #         list_algo_1.append(name_first)
            path_sec = '../data/'+env_type+'_'+str(algo_list[1])+'_'+param+'_'+problem+'/data.csv'
    #         name_no = '../data/ambulance_metric'+problem+'_'+param+'_No_Movement'+'/data.csv'
    #         list_algo_2.append(name_sec)

            path_third = '../data/'+env_type+'_'+str(algo_list[2])+'_'+param+'_'+problem+'/data.csv'

            fig_path = '../figures/'
            line_fig_name = '../figures/'+env_type+'_'+problem+'_'+param+'_line_plot'+'.pdf'
            radar_fig_name = '../figures/'+env_type+'_'+problem+'_'+param+'_radar_plot'+'.pdf'
            
            #lineplots
            plot_line_plots(path_first, path_sec , path_third, fig_path , line_fig_name)
            
            #radarplots
            # plot_radar_plots(path_first, path_sec , fig_path , radar_fig_name)
