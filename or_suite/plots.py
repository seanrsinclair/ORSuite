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
        print(max_value, min_value, feature_name)
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
                
    return result


'''

Create a set of line_plots for the algorithms, comparing the three metrics of reward, time, and space complexity

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