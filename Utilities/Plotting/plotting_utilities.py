import os
import seaborn as sns
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import ModelTraining.Utilities.Plotting.utils as plt_utils


def rename_columns(df, labels):
    return df.copy().rename({col: label for col, label in zip(list(df.columns), labels)}, axis=1) if labels else df


'''
Parameters:
- Simulation Results - Dict
- Plot path
- output file name
- labels
'''
def plot_result(data: pd.DataFrame, plot_path="./", output_file_name='Result', store_to_csv=True, **kwargs):
    if data is not None:
        fig, ax, = plt_utils.create_figure(output_file_name, figsize=kwargs.pop('figsize', None))
        plt.xlabel('Time')
        if kwargs.get('ylim',None):
            plt.set_ylim(kwargs.pop('ylim'))
        if kwargs.get('ylabel',None):
            plt.ylabel(kwargs.pop('ylabel'))
        plt_utils.plot_df(ax, data, **kwargs)
        plt_utils.save_figure(plot_path,output_file_name)
        if store_to_csv:
            data.to_csv(os.path.join(plot_path, f'{output_file_name}.csv'), index=True)



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


def printHeatMap(corr, dir="./", filename="Correlation", plot_enabled=True, annot=False):
    corr.to_csv(os.path.join(dir, f'{filename}.csv'))
    if plot_enabled:
        plt.figure(figsize=(15,15))
        sns.heatmap(corr, vmin=-1, vmax=1, cmap='coolwarm',annot=annot, fmt='.2f')
        plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
        plt_utils.save_figure(dir, filename, store_tikz=False)


'''
Parameters:
- DataFrame
'''
def scatterplot(y_pred, y_true, fig_save_path="./", filename="Scatterplot", **kwargs):
    pd.DataFrame(index=y_true.flatten(), data=y_pred, columns=['y']).to_csv(os.path.join(fig_save_path, f'{filename}.csv'), index_label='x')
    fig, ax, = plt_utils.create_figure(figsize=kwargs.pop('figsize', None), fig_title=kwargs.pop('fig_title',""))
    ax.scatter(y_true,y_pred,alpha=0.5,
                color=kwargs.get('color','blue'), label=kwargs.get('label'))
    #plt.title("Linear Regression model")
    ax.set_xlabel("True values")
    ax.set_ylabel("Predicted values")
    limits = [min(min(y_true),min(y_pred)),max(max(y_true),max(y_pred))]
    ax.plot([limits[0],limits[1]],[limits[0],limits[1]], color='k',linestyle='-', label="Optimal Prediction")
    #plt.xlim(limits)
    #plt.ylim(limits)
    ax.legend()
    plt_utils.save_figure(fig_save_path, filename)


import numpy as np
def barplot(feature_names, values, fig_save_path = './', filename='Features', **kwargs):
    fig_title = kwargs.pop('fig_title',"")
    fig, ax = plt_utils.create_figure(fig_title, figsize=kwargs.pop('figsize', None))
    plt.tight_layout(rect = [0.05,0.3,1.0,1])
    index = np.arange(len(feature_names))
    plt.bar(index, values)
    if kwargs.get('ylabel', None):
        plt.ylabel(kwargs.pop('ylabel'))
    ax.set_xticks(index, labels=feature_names, rotation=90)
    plt_utils.save_figure(fig_save_path, filename)


def barplot_df(df, fig_save_path = './', fig_title='Features', **kwargs):
    barplot(df.columns, df.values.flatten(), fig_save_path, fig_title, **kwargs)