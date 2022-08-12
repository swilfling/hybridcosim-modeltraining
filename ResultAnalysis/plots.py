import os
import matplotlib.pyplot as plt

import Plotting.plot_data as plt_utils
import tikzplotlib
from ModelTraining.ResultAnalysis.result_utils import get_result_df

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

    scatter_dir = f'{output_dir}/Scatter'
    timeseries_dir = f'{output_dir}/Timeseries'
    os.makedirs(timeseries_dir, exist_ok=True)
    os.makedirs(scatter_dir, exist_ok=True)

#%% Timeseries plots
    for usecase, target_val in zip(usecase_names, target_vals):
        model_types = ['WeightedLS'] if usecase == "SensorC6" else ['LassoRegression'] if usecase == "SensorA6" else ['RidgeRegression']

        for threshold_set in thresholds_rfvals:
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full)
            df = get_result_df(path, model_types, baseline_model_type, target_val, expansion)
            df.to_csv(os.path.join(timeseries_dir, f'{usecase}.csv'), index_label='t')

            plt.figure(figsize=(15,5))
            ylabel = 'Energy Consumption' if target_val == 'energy' else 'Gas Consumption' if target_val == 'B20Gas' else 'Solar Collector Supply Temperature'
            plt.title(f'{ylabel} - Dataset {usecase}')
            for color, column in zip(plot_colors,df.columns):
                plt.plot(df[column], linewidth=0.75, color=color)
            plt.legend(df.columns,loc='upper right')
            plt.ylabel(f'{ylabel} [kWh]')
            plt.xlabel('Time')
            plt.grid('both')
            tikzplotlib.save(f'{timeseries_dir}/{usecase}.tex')
            plt.savefig(f'{timeseries_dir}/{usecase}.png')
            plt.show()

            y_true = df['Measurement value']
            for label, color in zip(df.columns[1:],plot_colors[1:]):
                # Scatterplot
                y_pred = df[label]
                plt_utils.scatterplot(y_pred, y_true, scatter_dir,
                                   filename=f'Scatter_{usecase}_{thresh_name_full}_{label}'.replace(" ", ""),
                                   fig_title=f'Correlation - Dataset {usecase}',
                                   figsize=(4, 4), color=color, label=label)


