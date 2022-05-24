import matplotlib.pyplot as plt
import seaborn as sns
import ModelTraining.Utilities.Plotting.utils as plt_utils
import pandas as pd
import statsmodels.api as sm
import os


def plot_qq(data:pd.DataFrame, path, title, **kwargs):
    fig, ax = plt_utils.create_figure(fig_title=title)
    obj_probplot = sm.ProbPlot(data)
    df_qq = pd.DataFrame(index=obj_probplot.theoretical_quantiles, data=obj_probplot.sample_quantiles, columns=['y'])
    if kwargs.pop('store_csv',False):
        df_qq.to_csv(os.path.join(path, f'{title}.csv'),index_label='x')
    qq = obj_probplot.qqplot(marker='o', alpha=1, label='QQ', ax=ax)
    ax0 = qq.axes[0]
    sm.qqline(ax0, line='45', fmt='k--')
    plt.grid(visible=False)
    ax0.legend(loc='upper left')
    plt_utils.save_figure(path, title)
    plt.show()


def plot_density(data: pd.DataFrame, path, title, omit_zero_samples=False, **kwargs):
    if omit_zero_samples:
        data = [data[feature][data[feature] != 0] for feature in data.columns]
    g = sns.kdeplot(data=data, color='darkblue')
    ax = plt.gca()
    for line in ax.lines:
        xdata = line.get_xdata()
        ydata = line.get_ydata()
        df = pd.DataFrame(ydata, columns=['y'], index=xdata)
    df.to_csv(os.path.join(path, f'{title.replace(" ","")}.csv'), index_label='x')
    ax.set_title(title)
    plt.tight_layout()
    plt_utils.save_figure(path, title, **kwargs)
    plt.show()


'''
Parameters:
- Data - not necessarily data frame
- figure save path
- titles 
'''
def plot_histograms(data, fig_save_path, fig_title, titles):
    n = len(titles)
    fig, axes = plt.subplots(n,1)
    plt.tight_layout(h_pad=0)
    plt.suptitle(fig_title)

    fig.set_size_inches(7, n * 5)
    for i in range(n):
        cur_data = data.iloc[:,i]
        min = cur_data.min()
        max = cur_data.max()
        #print(min)
        #print(max)
        axes[i].hist(cur_data, range=(min,max))
        axes[i].set_title(titles[i])

    plt_utils.save_figure(fig_save_path, fig_title)
    plt.show()


'''
Parameters:
- List of variables
- DataFrame
- figure settings
- plot params
'''
def plot_missing_values(all_data,filename):
    '''Visualize the missigness'''
    # we need to change the index format otherwise seaborn gives UTC000000 length labels.
    #all_data_visual = all_data.set_index(all_data.index.strftime('%d.%m.%Y'))
    all_data_visual = all_data
    fig = plt.figure(figsize=(20,20))
    ax = sns.heatmap(all_data_visual.isna(), cbar=False)
    ax.set_ylabel('Time Period')
    ax.set_xlabel('Sensors')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=10)
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()



'''
Parameters:
- ax
- data
- color
- labels
'''
def hist_50bins(ax, data, color, xlabel="", ylabel=""):
    ax.hist(x=data, bins=50,
             range=(data.min(), data.max()),
             color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


'''
Parameters:
- data
- plot_ac
- spectrum range
'''
def plot_spectra(data, fft_data):
    # Plot FFT
    fig, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_xlabel('Time in Number of Samples')
    ax1.set_title("Time Series - Zero Mean")
    ax1.plot(data)
    ax2.set_title("Spectrum - Magnitude Response")
    ax2.stem(fft_data[0], fft_data[1])
    ax2.set_xlabel("Frequency normalized over sampling frequency f_s")
    plt.tight_layout()
    plt.show()


def plot_ac(xcorr, psd):
    fig2, (ax1, ax2) = plt.subplots(2,1)
    ax1.set_title("Auto-correlation - Normalized")
    ax1.plot(xcorr)
    ax2.set_title("Power Spectral Density")
    plt.plot(psd)
    plt.tight_layout()
    plt.show()
