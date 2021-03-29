import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib
import pickle
import numpy as np
import pandas as pd
from math import pi


from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D





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




def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta








def plot_radar_plots(path_list, algo_list, fig_path , fig_name, additional_metric):
    
    # Organizing Data

    plt.style.use('../or_suite/PaperDoubleFig.mplstyle.txt')
    plt.rc('text', usetex=True)




    # Generating the dataframe

    index = 0
    values = []
    for algo in algo_list:
        df = pd.read_csv(path_list[index]+'/data.csv').groupby(['episode']).mean()
        df['episode'] = df.index.values 
        df = df[df['episode'] == df.max()['episode']] # THIS IS NOT TOTALLY CORRECT, SHOULD BE SUM OVER EPISODES FOR TIME AND SPACE?

        algo_dict = {'Algorithm': algo, 'Reward': df.iloc[0]['epReward'], 'Time':df.iloc[0]['time'], 'Space': df.iloc[0]['memory']}


        with open(path_list[index]+'/trajectory.obj', 'rb') as f:
            x = pickle.load(f)
            for metric in additional_metric:
                algo_dict[metric] = additional_metric[metric](x)

        values.append(algo_dict)

        index += 1

    

    # Set data
    df = pd.DataFrame(values)

    print(df)
    df = normalize(df)



    # number of variable
    spoke_labels=list(df)[1:]
    N = len(spoke_labels)



    theta = radar_factory(N, frame='polygon')


    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(top=0.85, bottom=0.05)


    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title('Comparison of Performance Metrics',  position=(0.5, 1.1), ha='center')


    index = 0

    for algo in algo_list:
        values = df.loc[index].values[1:]
        ax.plot(theta, values, linewidth=1, linestyle='solid', label=algo, color = sns.color_palette('colorblind', len(algo_list))[index])
        ax.fill(theta, values, color = sns.color_palette('colorblind', len(algo_list))[index], alpha=0.25)
        index += 1

    ax.set_varlabels(spoke_labels)

    plt.legend(loc='right', bbox_to_anchor = (1.75 , .5))





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
def plot_line_plots(path_list, algo_list, fig_path , fig_name, plot_freq):
    plt.style.use('../or_suite/PaperDoubleFig.mplstyle.txt')
    plt.rc('text', usetex=True)

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
        df = df.iloc[::plot_freq, :]

        # PLOT OF OBSERVED REWARDS
        ax[0].plot(df['episode'], df['epReward'], label=algo, dashes = dash_styles[index], color = sns.color_palette('colorblind', len(algo_list))[index])

        ax[1].plot(df['episode'], df['time'], label=algo, dashes = dash_styles[index], color = sns.color_palette('colorblind', len(algo_list))[index])

        ax[2].plot(df['episode'], df['memory'], label=algo, dashes = dash_styles[index], color = sns.color_palette('colorblind', len(algo_list))[index])

        index += 1


    ax[0].set_ylabel('Observed Reward')
    ax[1].set_ylabel('Observed Time Used (log scale)')
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