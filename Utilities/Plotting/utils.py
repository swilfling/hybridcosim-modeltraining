import os
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import datetime
import tikzplotlib
import pandas as pd
'''
Parameters:
- plot dir
- output file name
'''
def save_figure(plot_path, output_file_name, format="png",store_tikz=True):
    filename = str(output_file_name).replace(" ", "_")
    plt.savefig(os.path.join(plot_path, f"{filename}.{format}"), format=format)
    if store_tikz:
        tikzplotlib.save(os.path.join(plot_path, f"{filename}.tex"))
    #plt.show()
'''
Parameters:
- output file name
- plotting variables
'''
def create_figure(fig_title="",**kwargs):
    fig = plt.figure(figsize=kwargs.pop('figsize',(20, 10)))
    #plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.0)
    plt.tight_layout()
    #plt.xlabel('Time')
    plt.grid('both')
    plt.suptitle(fig_title)
    ax = plt.gca()
    return fig, ax


'''
This function returns a str if param label is a list of length 1
'''
def label_list_to_str(labels):
    return ", ".join(label for label in labels) if type(labels) == list else str(labels)


def get_min(data):
    return data.min(skipna=True) if data.shape[1] == 1 else data.min(skipna=True)[0,0]


def get_max(data):
    return data.max(skipna=True) if data.shape[1] == 1 else data.max(skipna=True)[0,0]


def create_time_axis_days(index, divider=3600 * 24):
    return [val.total_seconds() / divider for val in index]

'''
Parameters:
- Simulation Results - Dict
- label names
- colors
- args
'''
def plot_df(ax: plt.Axes, simulation_results: pd.DataFrame, **kwargs):
    if simulation_results is not None:
        if kwargs.pop('show_legend',True):
            ax.legend(simulation_results.columns)
        if kwargs.pop('show_ylabel', False):
            ax.set_ylabel(label_list_to_str(list(simulation_results.columns)))
        if kwargs.pop('set_colors', False):
            ax.set_prop_cycle(kwargs.pop('cycler', None))
        if kwargs.get('xdate_format', None):
            #    simulation_results.index = pd.DatetimeIndex(pd.to_datetime(0) + simulation_results.index)
            #ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=range(int(mdates.DAYS_PER_YEAR))))
            ax.xaxis.set_major_formatter(mdates.DateFormatter(kwargs.pop('xdate_format')))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        if type(simulation_results.index) == pd.TimedeltaIndex:
            ax.plot(create_time_axis_days(simulation_results.index), simulation_results, **kwargs)
            ax.set_xlabel("Time [Days]")
        else:
            ax.plot(simulation_results, **kwargs)


def plot_df_twinx(ax, data, **kwargs):
    if type(data) == list and len(data) > 1:
        plot_df(ax, data[0])
        ax2 = plt.twinx()
        plot_df(ax2, data[1], **kwargs)
    else:
        df = data[0] if type(data) == list else data
        plot_df(ax, df, **kwargs)
