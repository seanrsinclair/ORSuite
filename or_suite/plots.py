import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib
import pickle
import numpy as np
import pandas as pd
from math import pi





'''

Normalizing the columns of a dataframe (except the first one which just contains algorithm names)

'''


def normalize(df):
    result = df.copy()
    for feature_name in df.columns[1:]:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
                
    return result




def plot_radar_plots(path_list, algo_list, fig_path , fig_name):
    
    # Organizing Data
    
    index = 0
    reward = []
    time = []
    space = []
    for algo in algo_list:
        df = pd.read_csv(path_list[index]).groupby(['episode']).mean()
        df['episode'] = df.index.values 
        df = df[df['episode'] == df.max()['episode']] # THIS IS NOT TOTALLY CORRECT, SHOULD BE SUM OVER EPISODES FOR TIME AND SPACE?
        reward.append(df.iloc[0]['epReward']) # TODO: Figure out how to get this to work with additional user-defined metrics
        time.append(df.iloc[0]['time'])
        space.append(df.iloc[0]['memory'])
        
        index += 1

    

    # Set data
    df = pd.DataFrame({'group': algo_list,
                       'Time': time,
                       'Space': space,
                       'Reward': reward})

    df = normalize(df)
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



    index = 0

    for algo in algo_list:
        values = df.loc[index].drop('group').values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=1, linestyle='solid', label=algo, color = sns.color_palette('colorblind', len(algo_list))[index])
        ax.fill(angles, values, color = sns.color_palette('colorblind', len(algo_list))[index], alpha=0.1)
        index += 1

    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon = True)



    if os.path.exists(fig_path):
                plt.savefig(os.path.join(fig_path,fig_name), bbox_inches = 'tight',
                pad_inches = 0.01)
    else:
                os.makedirs(fig_path)
                plt.savefig(os.path.join(fig_path,fig_name), bbox_inches = 'tight',
                pad_inches = 0.01)




'''

Create a set of line_plots for the algorithms, comparing the three metrics of reward, time, and space complexity

    Path_List: list of the paths to the folders containing the data.csv files
    Algo_List: list of the algorithm name
    Fig_Path: Path for the location to save the figure
    Fig_Name: name of the figure

'''
def plot_line_plots(path_list, algo_list, fig_path , fig_name):
    plt.style.use('../or_suite/PaperDoubleFig.mplstyle.txt')
    # plt.rc('text', usetex=True)

    fig, ax = plt.subplots(1, 3, constrained_layout=False, figsize=(15,5))
    fig.suptitle('Comparison of Performance Metrics')


    dash_styles = ["",
               (4, 1.5),
               (1, 1),
               (3, 1, 1.5, 1),
               (5, 1, 1, 1),
               (5, 1, 2, 1, 2, 1),
               (2, 2, 3, 1.5),
               (1, 2.5, 3, 1.2)]

    filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')


    index = 0

    for algo in algo_list:
        df = pd.read_csv(path_list[index]).groupby(['episode']).mean()
        df['episode'] = df.index.values

        # PLOT OF OBSERVED REWARDS
        ax[0].plot(df['episode'], df['epReward'], label=algo, dashes = dash_styles[index], color = sns.color_palette('colorblind', len(algo_list))[index])

        ax[1].plot(df['episode'], df['time'], label=algo, dashes = dash_styles[index], color = sns.color_palette('colorblind', len(algo_list))[index])

        ax[2].plot(df['episode'], df['memory'], label=algo, dashes = dash_styles[index], color = sns.color_palette('colorblind', len(algo_list))[index])



        index += 1


    ax[0].set_ylabel('Observed Reward')
    ax[1].set_ylabel('Observed Time Used ( ns )')
    ax[2].set_ylabel('Observed Memory Usage (B)')

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