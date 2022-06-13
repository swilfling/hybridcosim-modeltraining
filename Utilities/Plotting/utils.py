"""
Plotting utility functions
Includes:
 - Create figure
 - Plot dataframe
 - Save figure

"""

import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import tikzplotlib
import pandas as pd


def create_figure(fig_title="", figsize=(20, 10), **kwargs):
    """
    Create figure
    @param fig_title: Title
    @param figsize: figure size
    Optional params - not used right now
    """
    fig = plt.figure(figsize=figsize)
    plt.tight_layout()
    plt.grid('both')
    plt.suptitle(fig_title)
    ax = plt.gca()
    return fig, ax


def plot_df(ax: plt.Axes, data: pd.DataFrame, **kwargs):
    """
    Plot dataframe
    @param ax: axis to plot on
    @param data: pandas dataframe
    Optional parameters: show_legend, show_ylabel, set_colors, cycler, xdate_format, time_divider
    """
    if data is not None:
        show_legend = kwargs.pop('show_legend', True)
        if kwargs.pop('show_ylabel', False):
            label_str = ", ".join(label for label in data.columns)
            ax.set_ylabel(label_str)
        if kwargs.pop('set_colors', False):
            ax.set_prop_cycle(kwargs.pop('cycler', None))
        if kwargs.get('xdate_format', None):
            ax.xaxis.set_major_formatter(mdates.DateFormatter(kwargs.pop('xdate_format')))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if type(data.index) == pd.TimedeltaIndex:
            timedelta_divider = kwargs.pop('time_divider', 3600 * 24)
            ax.plot([val.total_seconds() / timedelta_divider for val in data.index], data, **kwargs)
            ax.set_xlabel("Time [Days]")
        else:
            ax.plot(data, **kwargs)
        if show_legend:
            ax.legend(data.columns)


def save_figure(path="./", filename="Plot", fmt="png", store_tikz=True):
    """
    Save figure - Optional: save to tikz
    @param path: Directory to store to
    @param filename: Output filename
    @param fmt: File format
    @param store_tikz: store as tikz .tex file
    """
    filename = str(filename).replace(" ", "_")
    plt.savefig(os.path.join(path, f"{filename}.{fmt}"), format=fmt)
    if store_tikz:
        tikzplotlib.save(os.path.join(path, f"{filename}.tex"))



