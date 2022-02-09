import os
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import Utilities.Plotting.utils as plt_utils


def rename_columns(df, labels):
    return df.copy().rename({col: label for col, label in zip(list(df.columns), labels)}, axis=1) if labels else df


'''
Parameters:
- Simulation Results - Dict
- Plot path
- output file name
- labels
'''
def plot_result(data, plot_path, output_file_name, store_to_csv=False, **kwargs):
    fig, ax = plt_utils.create_figure(output_file_name, **kwargs)
    if kwargs.get('ylim',None):
        plt.set_ylim(kwargs.pop('ylim'))
    plt_utils.plot_df(ax, data, **kwargs)
    plt_utils.save_figure(plot_path,output_file_name)
    plt.show()
    if store_to_csv:
        data.to_csv(os.path.join(plot_path, f'{output_file_name}.csv'), index=True)


'''
Parameters:
- List of Dicts
- path
- output file name
- labels
'''
def plot_multiple_results(list_simulation_results: List[pd.DataFrame], plot_path, output_file_name, set_colors=False, **kwargs):
    if len(list_simulation_results) > 0:
        colors = kwargs.pop('colors', None)
        fig, ax = plt_utils.create_figure(output_file_name, **kwargs)
        linestyles=["--" if i % 2 else "-" for i in range(len(list_simulation_results))]
        for simulation_results, linestyle in zip(list_simulation_results, linestyles):
            cycler = plt.cycler(color=sns.color_palette(colors, simulation_results.shape[1]))
            plt_utils.plot_df(ax,simulation_results, linestyle=linestyle, set_colors=set_colors, cycler=cycler, show_ylabel=True)
        plt_utils.save_figure(plot_path, output_file_name)
        plt.show()


'''
Parameters:
- List of variables
- DataFrame
- Title
- output dir
'''
def plt_subplots(list_df : List, fig_save_path, fig_title, **kwargs):
    fig, ax = plt_utils.create_figure(fig_title, figsize=kwargs.pop('figsize', None))
    if kwargs.get('ylim',None):
        plt.ylim(kwargs.pop('ylim'))
    for index, data in enumerate(list_df):
        ax = plt.subplot(len(list_df), 1, index + 1)
        plt.grid('on')
        plt_utils.plot_df_twinx(ax, data, **kwargs)
    fig.suptitle(fig_title)
    plt_utils.save_figure(fig_save_path, fig_title)
    plt.show()



def plot_df_subplots(df,fig_save_path, fig_title, plot_params={'axes.labelsize':8,
                            'axes.titlesize':8,
                            'font.size':6,
                            'font.weight':'normal',
                            'xtick.labelsize':6,
                            'ytick.labelsize':6,
                            #'figure.figsize':[7.3,4.2],
                            'figure.figsize':[10,10],
                            'axes.linewidth':1,
                            'xtick.major.size':6,
                            'ytick.major.size':6}, **kwargs):
    #      rcParams.update(plot_params)
    fig, ax = plt_utils.create_figure(fig_title, figsize=kwargs.pop('figsize', None))
    df.plot(subplots=True, ax=ax)
    fig.suptitle(fig_title)
    plt_utils.save_figure(fig_save_path, fig_title)
    plt.show()


def printHeatMap(dataframe):
    label_encoder = LabelEncoder()
    dataframe.iloc[:,0] = label_encoder.fit_transform(dataframe.iloc[:,0]).astype('float64')
    corr = dataframe.corr()
    sns.heatmap(corr)
    plt.show()


'''
Parameters:
- DataFrame
'''
def scatterplot(y_pred, y_true, legend=None, **kwargs):
    fig, ax = plt_utils.create_figure('Correlation', figsize=kwargs.pop('figsize', None))
    ax.scatter(y_true,y_pred,alpha=0.5,
                color='blue', label=legend)
    #plt.title("Linear Regression model")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    limits = [min(min(y_true),min(y_pred)),max(max(y_true),max(y_pred))]
    ax.plot([limits[0],limits[1]],[limits[0],limits[1]], color='k',linestyle='-', label="Optimal Prediction")
    #plt.xlim(limits)
    #plt.ylim(limits)
    ax.legend()
    plt.show()

