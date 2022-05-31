#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import ModelTraining.Utilities.Plotting.plotting_utilities as plt_utils


if __name__ == "__main__":
    # %%
    usecase_names = ['CPS-Data','SensorA6', 'SensorB2', 'SensorC6', 'Solarhouse1', 'Solarhouse2']
    target_vals = ['energy'] * 4 + ['TSolarVL','T_Solar_VL']
    solarhouse_usecases = ['Solarhouse1','Solarhouse2']
    metrics = ['R2', 'CV-RMS', 'MAPE']

    thresholds_rfvals = [['MIC-value_0.05','R-value_0.05']]
    expansion = [['IdentityExpander','IdentityExpander'], ['IdentityExpander','PolynomialExpansion']]

    model_types = ['RidgeRegression']
    timestamp = '20220421_145209'
    experiment_name = f'Experiment_{timestamp}'
    result_dir = os.path.join('../', 'results', experiment_name)
    output_dir = f"./Figures/{experiment_name}"
    timestamp = "20220414_160549"

    #%% Linreg coeffs
    coeff_dir = os.path.join(output_dir, 'ModelProperties')
    os.makedirs(coeff_dir,exist_ok=True)
    for usecase, target_val in zip(usecase_names, target_vals):
        for threshold_set, expansion_set in zip(thresholds_rfvals, expansion):
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'ModelProperties')
            expansion_name = expansion_set[-1]
            print(expansion_name)
            for model_type in model_types:
                df = pd.read_csv(os.path.join(path, f'Coefficients_{model_type}_{target_val}_{expansion_name}.csv'))
                df = df.set_index(df.columns[0])
                if usecase in solarhouse_usecases:
                    figsize = (6,4) if expansion_name == 'IdentityExpander' else (10,10)
                else:
                    figsize = (8,6) if expansion_name == 'IdentityExpander' else (30,10)

                plt_utils.barplot(df.index, df.values.flatten(), coeff_dir,
                                  filename=f'Coefficients_{model_type}_{target_val}_{usecase}_{thresh_name_full}_{expansion_name}',
                                  fig_title=f"{model_type} Coefficients - {target_val} - Dataset {usecase}", figsize=figsize,
                                  ylabel='Coefficient')


    #%% Metrics
    metrics_dir = os.path.join(output_dir, 'Metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    for metric in metrics:
        results_file = os.path.join(result_dir, f'Metrics_{timestamp}_{metric}.csv')
        df = pd.read_csv(results_file)
        df.set_index('Model')

        plt.figure(figsize=(8,6))
        width=1
        x = np.arange(len(df.columns)-1)
        plt.bar(x,df.iloc[-1].values[1:],label='Random Forest', color='lightgray')
        plt.title(f'Metrics - {metric}')
        for i in [0,1,3,4,5, 7]:
            vals = df.iloc[i].values[1:]
            plt.plot(vals, label=df.iloc[i,0])
            plt.bar(x+width*i, vals, width, label=df.iloc[i,0])
        plt.xticks(range(len(df.columns)-1),df.columns[1:])
        plt.grid('both')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(os.path.join(metrics_dir,f'Metrics_{metric}.png'))
        plt.show()


#%% Metrics - adjust file
    results_file = os.path.join(result_dir, f'Metrics_full_Experiment_{timestamp}.csv')
    df = pd.read_csv(results_file)
    maxcol = df.max(axis=0).astype('string')
    mincol = df.min(axis=0).astype('string')
    maxcol_sec = df[df != df.max(axis=0)].max(axis=0).astype('string')
    mincol_sec = df[df != df.min(axis=0)].min(axis=0).astype('string')
    df = df.astype('string')
    for col in df.columns[1:]:
        if 'R2' in col or 'RA' in col:
            df[col][df[col] == maxcol[col]] = f"\\textbf{{{maxcol[col]}}}"
            df[col][df[col] == maxcol_sec[col]] = f"\\textit{{{maxcol_sec[col]}}}"
        else:
            df[col][df[col] == mincol[col]] = f"\\textbf{{{mincol[col]}}}"
            df[col][df[col] == mincol_sec[col]] = f"\\textit{{{mincol_sec[col]}}}"
    print(df)
    df.to_csv(os.path.join(metrics_dir,f'Metrics_full_Experiment_{timestamp}_edited.csv'))


