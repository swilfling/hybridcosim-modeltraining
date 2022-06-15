"""
Plot distributions
    This file contains functions for:
    - Q-Q plot
    - Kernel density plot
    - Missing value plot
    - Heatmap
    - Histogram (multiple histograms as subplots)
"""
import seaborn as sns
from matplotlib import pyplot as plt
from . import utils as plt_utils
import pandas as pd
import statsmodels.api as sm
import os


def plot_qq(data: pd.Series, path="./", filename="Q-Q", **kwargs):
    """
    Plot Q-Q plot of set of vals, stores plot as png (tikz optional) and q-q vals as csv (optional)
    @param data: pd.Series of values
    @param path: output directory for CSV
    @param filename: plot title - is used as output file name
    Optional parameters: xlim, ylim, store_csv, fig_title
    """
    title = kwargs.pop('fig_title', "")
    fig, ax = plt_utils.create_figure(fig_title=title)
    ylim = kwargs.pop('ylim', None)
    xlim = kwargs.pop('xlim', None)
    obj_probplot = sm.ProbPlot(data)
    # Store to csv
    df_qq = pd.DataFrame(index=obj_probplot.theoretical_quantiles, data=obj_probplot.sample_quantiles, columns=['y'])
    if kwargs.pop('store_csv', False):
        df_qq.to_csv(os.path.join(path, f'{filename}.csv'), index_label='x')
    # Plot Q-Q and qqline
    qq = obj_probplot.qqplot(marker='o', alpha=1, label='QQ', ax=ax)
    ax0 = qq.axes[0]
    sm.qqline(ax0, line='45', fmt='k--')
    # Grid and legend
    plt.grid(visible=False)
    plt.ylim(ylim)
    plt.xlim(xlim)
    ax0.legend(loc='upper left')
    # Save figure
    plt_utils.save_figure(path, filename)
    plt.show()


def plot_density(data: pd.DataFrame, path="./", filename="Density", omit_zero_samples=False, **kwargs):
    """
    Plot kernel density estimation - store figure (tikz optional), store values in csv
    @param data: dataframe containing values
    @param path: output directory
    @param filename: Filename
    @param omit_zero_samples: Omit zero samples in density plot
    Optional parameters: store_tikz, fig_title, arguments for kdeplot
    """
    store_tikz = kwargs.pop('store_tikz', False)
    title = kwargs.pop('fig_title', "")
    feature_names = data.columns
    # Omit zero samples - adjust data
    if omit_zero_samples:
        data = [data[feature][data[feature] != 0] for feature in data.columns]
    # Plot density
    ax = plt.gca()
    for feature in data:
        ax = feature.plot.kde(**kwargs)
        line = ax.lines[-1]
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        # Store to csv
        df = pd.DataFrame(ydata, columns=['Density'], index=xdata)
        df.to_csv(os.path.join(path, f'{filename}_{feature.name}_.csv'), index_label='x')
    ax.legend(feature_names)
    ax.set_title(title)
    plt.tight_layout()
    plt_utils.save_figure(path, filename, store_tikz=store_tikz)
    plt.show()


def plot_missing_values(data: pd.DataFrame, path="./", filename="Density", **kwargs):
    """
    Plot missing values as heatmap
    @param data: pd.Dataframe
    @param path: output directory
    @param filename: figure filename
    """
    printHeatMap(data.isna(), path, filename, cbar=False, **kwargs)


def printHeatMap(data: pd.DataFrame, path="./", filename="Correlation", annot=False, **kwargs):
    """
    Plot heatmap
    @param data: pd.Dataframe
    @param path: output directory
    @param filename: figure title
    @param annot: annote heatmap or not
    Optional args: figsize, cmap, vmin, vmax, store_tikz, cmap, fig_title
    """
    # Store data to csv
    data.to_csv(os.path.join(path, f'{filename}.csv'))
    store_tikz = kwargs.pop('store_tikz', False)
    # Plot heatmap
    plt_utils.create_figure(fig_title=kwargs.pop('fig_title', ""), figsize=kwargs.pop('figsize', (15,15)))
    sns.heatmap(data, annot=annot, **kwargs)
    plt.tight_layout(rect=[0.1, 0.1, 0.9, 0.9])
    plt_utils.save_figure(path, filename, store_tikz=store_tikz)
    plt.show()


def plot_histograms(data: pd.DataFrame, path="./", filename="Histograms", subplot_titles=None, **kwargs):
    """
    Plot histograms as subplots
    @param data: dataframe containing values
    @param path: output directory
    @param title: figure title
    @param subplot_titles: list of subplot titles - optional
    Optional: fig_title
    """
    n = len(data.columns)
    fig, axes = plt.subplots(n)
    if subplot_titles is None:
        subplot_titles = data.columns
    plt.tight_layout(h_pad=0)
    plt.suptitle(kwargs.pop('fig_title', ""))
    fig.set_size_inches(7, n * 5)
    for subtitle, ax, col in zip(subplot_titles, axes, data.columns):
        ax.hist(data[col], range=(data[col].min(),data[col].max()))
        ax.set_title(subtitle)
    plt_utils.save_figure(path, filename)
    plt.show()


