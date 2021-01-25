import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib
import pickle
import numpy as np


import pandas as pd

epLen = 5
nEps = 1000



problem_type = 'ambulance'
problem_list = ['shifting', 'beta', 'uniform']
param_list = ['0', '1', '25']


for problem in problem_list:
    for param in param_list:

        name_median = '../data/ambulance_'+problem+'_'+param+'_Median'+'.csv'
        name_no = '../data/ambulance_'+problem+'_'+param+'_No_Movement'+'.csv'

        fig_name = '../figures/ambulance_'+problem+'_'+param+'.eps'




        dt_median = pd.read_csv(name_median).groupby(['episode']).mean()
        dt_no = pd.read_csv(name_no).groupby(['episode']).mean()
        dt_median['episode'] = dt_median.index.values
        dt_no['episode'] = dt_no.index.values
        dt_no = dt_no.iloc[::10, :]
        dt_median = dt_median.iloc[::10, :]


        fig = plt.figure(figsize=(3.5, 3.5))
        plt.title('Comparison of Observed Rewards')
        plt.plot(dt_median['episode'], dt_median['epReward'], label='Median', linestyle=':')
        plt.plot(dt_no['episode'], dt_no['epReward'], label = 'No Movement', linestyle='-.')

        plt.ylim(0,epLen+.1)
        plt.xlabel('Episode')
        plt.ylabel('Observed Reward')
        plt.legend()

        plt.tight_layout()
        fig.savefig(fig_name, bbox_inches = 'tight',
            pad_inches = 0.01, dpi=900)
