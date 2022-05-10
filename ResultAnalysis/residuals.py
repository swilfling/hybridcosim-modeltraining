
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ModelTraining.Utilities.Plotting.plotting_utilities as plt_utils
import statsmodels.api as sm

#%%

def env_max(maxvals, windowsize):
    maxvals_downsampled = maxvals.rolling(window=windowsize).max().drop_duplicates()
    maxvals_downsampled[maxvals.index[0]] = maxvals.iloc[0]
    maxvals_downsampled[maxvals.index[-1]] = maxvals.iloc[-1]
    return maxvals_downsampled.reindex(maxvals.index).interpolate()

def env_min(maxvals, windowsize):
    maxvals_downsampled = maxvals.rolling(window=windowsize).min().drop_duplicates()
    maxvals_downsampled[maxvals.index[0]] = maxvals.iloc[0]
    maxvals_downsampled[maxvals.index[-1]] = maxvals.iloc[-1]
    return maxvals_downsampled.reindex(maxvals.index).interpolate()

#%%
def get_df(path, model_types, baseline_model_type, target_val, expansion):
    dict_expansion_names = {'IdentityExpander': 'basic features', 'PolynomialExpansion': 'Polyfeatures'}
    df = pd.DataFrame()
    for model_type in model_types:
        for expansion_set in expansion:
            df_new = pd.read_csv(
                os.path.join(path, f'Timeseries_{model_type}_{target_val}_{expansion_set[-1]}.csv'))
            if df.empty:
                df.index = df_new.index
                df['Measurement value'] = df_new[target_val]
            df[f'{model_type} - {dict_expansion_names[expansion_set[-1]]}'] = df_new[f'predicted_{target_val}']
    df[f'{baseline_model_type}'] = \
    pd.read_csv(os.path.join(path, f'Timeseries_{baseline_model_type}_{target_val}_IdentityExpander.csv'))[
        f'predicted_{target_val}']
    return df


if __name__ == "__main__":
# %%

    csv_files = ['CPS-Data', 'Sensor A6', 'Sensor B2', 'Sensor C6', 'Solarhouse 1', 'Solarhouse 2', 'Beyond B20 Gas']

    colormap = plt.cm.get_cmap('tab10')
    plot_colors = [colormap(i) for i in range(6)]

    experiment_name = 'Experiment_20220509_113409'
    result_dir = os.path.join('../', 'results', experiment_name)
    usecase_names = ['CPS-Data', 'SensorA6', 'SensorB2', 'SensorC6', 'Solarhouse1', 'Solarhouse2']
    target_vals = ['energy'] * 4 + ['TSolarVL', 'T_Solar_VL']
    units = ['kWh'] * 4 + ['°C'] * 2
    ylabels = ['Energy Consumption'] * 4 + ['Collector Supply Temperature'] * 1

    thresholds_rfvals = [['MIC-value_0.05', 'R-value_0.05']]
    expansion = [['IdentityExpander', 'IdentityExpander'], ['IdentityExpander', 'PolynomialExpansion']]

    model_types = ['RidgeRegression']
    baseline_model_type = 'RandomForestRegression'

    output_dir = f'./Figures/{experiment_name}'
    os.makedirs(output_dir, exist_ok=True)

    resid_dir = f'{output_dir}/Residuals'
    os.makedirs(resid_dir, exist_ok=True)
    resid_scatter_dir = f'{resid_dir}/Scatter'
    os.makedirs(resid_scatter_dir, exist_ok=True)
    resid_pp_dir = f'{resid_dir}/PP'
    os.makedirs(resid_pp_dir, exist_ok=True)

    scatter_dir = f'{output_dir}/Scatter'
    timeseries_dir = f'{output_dir}/Timeseries'
    os.makedirs(timeseries_dir, exist_ok=True)
    os.makedirs(scatter_dir, exist_ok=True)


#%% Timeseries plots
    for usecase, target_val in zip(usecase_names, target_vals):
        for threshold_set in thresholds_rfvals:
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'Plots')
            df = get_df(path, model_types, baseline_model_type, target_val, expansion)

            plt.figure(figsize=(15,5))
            ylabel = 'Energy Consumption' if target_val == 'energy' else 'Gas Consumption' if target_val == 'B20Gas' else 'Solar Collector Supply Temperature'
            plt.title(f'{ylabel} - Dataset {usecase}')
            for color,column in zip(plot_colors,df.columns):
                plt.plot(df[column], linewidth=0.75, color=color)
            plt.legend(df.columns,loc='upper right')
            plt.ylabel(f'{ylabel} [kWh]')
            plt.xlabel('Time')
            plt.grid('both')
            plt.savefig(f'{timeseries_dir}/{usecase}.png')
            #plt.show()

            y_true = df['Measurement value']

            for label, color in zip(df.columns[1:],plot_colors[1:]):
                # Scatterplot
                y_pred = df[label]
                plt_utils.scatterplot(y_pred, y_true, scatter_dir,
                                      filename=f'Scatter_{usecase}_{thresh_name_full}_{label}',
                                      fig_title=f'Correlation - Dataset {usecase}',
                                      figsize=(4,4), color=color, label=label)


#%% Residuals plots
    for usecase, target_val in zip(usecase_names, target_vals):
        for threshold_set in thresholds_rfvals:
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'Plots')
            df = get_df(path, model_types, baseline_model_type, target_val, expansion)
            # Residuals plot
            plt.figure(figsize=(15,5))
            for label,color in zip(df.columns[1:],plot_colors[1:]):
                plt.plot(df['Measurement value'] - df[label], label=f'Residual {label}', linewidth=0.75,color=color)
            plt.legend(loc='upper right')
            plt.ylabel(f'{ylabel} [kWh]')
            plt.xlabel('Time')
            plt.title(f'Residuals - Dataset {usecase} - {thresh_name_full}')
            plt.grid('both')
            plt.tight_layout()
            plt.savefig(f'{resid_dir}/Residuals_{usecase}_{thresh_name_full}.png')
            plt.show()


#%% Residuals P-P
    for usecase, target_val in zip(usecase_names, target_vals):
        for threshold_set in thresholds_rfvals:
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'Plots')
            df = get_df(path, model_types, baseline_model_type, target_val, expansion)
            y_true = df['Measurement value']

            # Residuals - Scatter
            for label,color in zip(df.columns[1:],plot_colors[1:]):#
                y_pred = df[label]
                residual = y_true - y_pred
                residual = (residual - np.mean(residual)) / np.std(residual)

                # P-P Plot
                fig = plt.figure()
                obj_probplot = sm.ProbPlot(residual)
                #obj_probplot.qqplot(line='45', ax=ax2, fmt='g',label='Q-Q')
                qq = obj_probplot.qqplot(marker='.', alpha=1, label='QQ')
                ax0 = qq.axes[0]
                ax1 = ax0.twinx()
                sm.qqline(ax0, line='45', fmt='k--')
                obj_probplot.probplot(ax=ax1, marker='.', markerfacecolor='g', markeredgecolor='g', label='PP')
                ax0.legend(loc='lower right')
                plt.legend(loc='upper left')

                plt.title(f'Dataset {usecase} - Residual - {label}')
                plt.savefig(f'{resid_pp_dir}/QQ_Residual_{usecase}_{thresh_name_full}_{label}.png')
                plt.show()




#%% Residuals Scatter
    for usecase, target_val in zip(usecase_names, target_vals):
        for threshold_set in thresholds_rfvals:
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'Plots')
            df = get_df(path, model_types, baseline_model_type, target_val, expansion)
            y_true = df['Measurement value']

            # Residuals - Scatter
            for label,color in zip(df.columns[1:],plot_colors[1:]):#
                y_pred = df[label]
                residual = y_true - y_pred
                residual = (residual - np.mean(residual)) / np.std(residual)

                y_pred_sorted = y_pred.drop_duplicates().sort_values()
                num_samples = len(residual)
                maxvals = pd.Series(index=y_pred_sorted, name='max')
                minvals = pd.Series(index=y_pred_sorted, name='min')
                for val in y_pred_sorted:
                    maxvals[val] = np.max(residual[y_pred == val])
                    minvals[val] = np.min(residual[y_pred == val])

                minvals_downsampled = env_min(minvals, round(num_samples / 30))
                maxvals_downsampled = env_max(maxvals, round(num_samples / 30))

                fig = plt.figure(figsize=(6,6))
                ax = plt.gca()
                plt.title(f'Dataset {usecase} {thresh_name_full}')
                plt.grid('both')
                ax.fill_between(y_pred_sorted,maxvals_downsampled, minvals_downsampled, where=(maxvals_downsampled > minvals_downsampled),alpha=0.7, color='lightgray',interpolate=True)
                ax.scatter(y_pred,residual,alpha=0.5,
                            color=color, label=label)
                ax.set_xlabel("Predicted values")
                ax.set_ylabel("Residual - Standardized")
                #plt.plot(maxvals_downsampled, color='lightgray')
                meanval =pd.DataFrame(data=[maxvals_downsampled, minvals_downsampled]).mean(axis=0)
                plt.plot(meanval, color='k', label='Residual Trend')
                #plt.plot(minvals_downsampled, color='lightgray')
                ax.plot([(min(y_pred)),max(y_pred)],[0,0], color='dimgray',linestyle='-', label="Optimal Prediction")
                plt.xlim([min(y_pred),max(y_pred)])
                plt.tight_layout()
                ax.legend()
                plt.savefig(f'{resid_scatter_dir}/Scatter_Residual_{usecase}_{thresh_name_full}_{label}.png')
                plt.show()





#%% Residuals Histogram
    resid_hist_dir = f'{resid_dir}/Hist'
    os.makedirs(resid_hist_dir,exist_ok=True)
    for usecase, target_val in zip(usecase_names, target_vals):
        for threshold_set in thresholds_rfvals:
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'Plots')
            df = get_df(path, model_types, baseline_model_type, target_val, expansion)
            y_true = df['Measurement value']
            # Residuals - Histogram
            for label,color in zip(df.columns[1:],plot_colors[1:]):
                y_pred = df[label]
                residual = y_true - y_pred
                residual = (residual - np.mean(residual)) / np.std(residual)
                # Histogram
                plt.figure(figsize=(8,4))
                n, bins, patches = plt.hist(residual, color=color, bins=2*round(np.sqrt(len(residual))), density=True, label=f'{label}')
                plt.text(np.min(bins), np.mean(n), f'Std dev: {np.round(np.std(residual),2)}')
                plt.plot([0,0],[0,np.max(n)*1.1], color='k',linestyle='-', label="Optimal Prediction")
                plt.plot([np.average(residual),np.mean(residual)],[0,np.max(n)*1.1], color='dimgray',linestyle='-', label="Mean value of residual")
                plt.title(f'Histogram - Residual - Dataset {usecase} - {thresh_name_full}')
                plt.ylim([0, 1.1*np.max(n)])
                plt.xlim([np.min(residual),np.max(residual)])
                plt.xlabel('Residual - Standardized')
                plt.ylabel('Density')
                plt.legend()
                plt.grid('both')
                plt.tight_layout()
                plt.savefig((f'{resid_hist_dir}/Hist_Residual_{usecase}_{thresh_name_full}_{label}.png'))
                plt.show()

#%%
    resid_spec_dir = f'{resid_dir}/Spectrum'
    os.makedirs(resid_spec_dir,exist_ok=True)
    for usecase, target_val in zip(usecase_names, target_vals):
        for threshold_set in thresholds_rfvals:
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'Plots')
            df = get_df(path, model_types, baseline_model_type, target_val, expansion)
            y_true = df['Measurement value']

            for label,color in zip(df.columns[1:],plot_colors[1:]):
                y_pred = df[label]
                residual = y_true - y_pred
                residual = (residual - np.mean(residual)) / np.std(residual)

                spec = np.abs(np.fft.fft(residual - np.mean(residual))/ len(residual))[:round(len(residual)/2)]
                # spectrum
                plt.figure(figsize=(10,5))
                d = 3600/4 if target_val == target_vals[-1] or target_val == target_vals[-2] else 3600
                plt.stem(24*3600*np.fft.fftfreq(len(residual), d=3600)[:round(len(residual)/2)],spec, label=f'Residual - {label}')
                plt.plot([1/365,1/365], [0,1.1*max(spec)], color='yellow', label = 'Yearly Trend')
                plt.plot([1/30,1/30], [0,1.1*max(spec)], color='orange', label = 'Monthly Trend')
                plt.plot([1/7,1/7], [0,1.1*max(spec)], color='red', label = 'Weekly Trend')
                plt.plot([1,1], [0,1.1*max(spec)], color='k', label = 'Daily Trend')
                plt.ylim([0,1.1*max(spec)])
                plt.gca().set_xscale('log')
                plt.ylabel('Magnitude Spectrum - normalized')
                plt.xlabel('Frequency logarithmic [1/d]')
                plt.legend()
                plt.title(f'Dataset {usecase} - Residual - Zero mean - {thresh_name_full}')
                plt.grid('both')
                plt.tight_layout()
                plt.savefig(f'{resid_spec_dir}/Spectrum_Residual_{usecase}_{thresh_name_full}_{label}_log.png')
                plt.show()

                plt.figure(figsize=(10,5))
                d = 3600/4 if target_val == target_vals[-1] or target_val == target_vals[-2] else 3600
                plt.stem(24*3600*np.fft.fftfreq(len(residual), d=d)[:round(len(residual)/2)],spec, label=f'Residual - {label}')
                plt.plot([1,1], [0,1.1*max(spec)], color='k', label = 'Daily Trend')
                if d == 3600/4:
                    plt.plot([6,6], [0,1.1*max(spec)], color='magenta', label = '4h Trend')
                    plt.plot([12,12], [0,1.1*max(spec)], color='mediumvioletred', label = '2h Trend')
                    plt.plot([24,24], [0,1.1*max(spec)], color='red', label = 'Hourly Trend')
                else:
                    plt.plot([2,2], [0,1.1*max(spec)], color='purple', label = '12h Trend')
                    #plt.plot([3,3], [0,1.1*max(spec)], color='darkviolet', label = '8h Trend')
                    plt.plot([4,4], [0,1.1*max(spec)], color='violet', label = '6h Trend')
                    plt.plot([6,6], [0,1.1*max(spec)], color='magenta', label = '4h Trend')
                    plt.plot([8,8], [0,1.1*max(spec)], color='pink', label = '3h Trend')
                    plt.plot([12,12], [0,1.1*max(spec)], color='mediumvioletred', label = '2h Trend')

                plt.ylim([0,1.1*max(spec)])
                plt.ylabel('Magnitude Spectrum - normalized')
                plt.xlabel('Frequency [1/d]')
                plt.legend()
                plt.title(f'Dataset {usecase} - Residual - Zero mean - {thresh_name_full}')
                plt.grid('both')
                plt.tight_layout()
                plt.savefig(f'{resid_spec_dir}/Spectrum_Residual_{usecase}_{thresh_name_full}_{label}.png')
                plt.show()
