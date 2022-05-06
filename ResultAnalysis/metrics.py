#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import tikzplotlib
from ModelTraining.Preprocessing.FeatureSelection.FeatureSelector import FeatureSelector
import ModelTraining.Utilities.Plotting.plotting_utilities as plt_utils


if __name__ == "__main__":
    # %%
    result_dir = os.path.join('../','results')
    usecase_names = ['CPS-Data','SensorA6', 'SensorB2', 'SensorC6', 'Solarhouse1', 'Solarhouse2']
    target_vals = ['energy'] * 4 + ['TSolarVL','T_Solar_VL']
    solarhouse_usecases = ['Solarhouse1','Solarhouse2']

    thresholds_rfvals = [['MIC-value_0.05','R-value_0.05']]
    expansion = [['IdentityExpander','IdentityExpander'], ['IdentityExpander','PolynomialExpansion']]

    model_types = ['RidgeRegression']
    timestamp = '20220421_145209'
    metrics = ['R2', 'CV-RMS', 'MAPE']

#%% Fvals

    os.makedirs('./Figures/RFvals',exist_ok=True)
    for usecase in usecase_names:
        for threshold_set, expansion_set in zip(thresholds_rfvals, expansion):
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'FeatureSelection')
            for threshold, expansion_name in zip(threshold_set, expansion_set):
                threshold_name = threshold.split("_")[0]
                selector_name = FeatureSelector.from_name(threshold_name).__name__
                df = pd.read_csv(os.path.join(path, f'Coefficients_{selector_name} - {expansion_name}.csv'), index_col='Feature')

                if usecase in solarhouse_usecases:
                    figsize = (5,5) if expansion_name == 'IdentityExpander' else (6,6)
                else:
                    figsize = (8,4) if expansion_name == 'IdentityExpander' else (20,15)

                plt_utils.barplot(df.index, df.values.flatten(), "./Figures/RFvals",filename=f'{threshold}_{usecase}_{thresh_name_full}_{expansion}',
                                  fig_title=f"{threshold_name} - Dataset {usecase}", figsize=figsize, ylabel=threshold_name)

#%% Linreg coeffs

    os.makedirs('./Figures/Coeffs',exist_ok=True)
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

                plt_utils.barplot(df.index, df.values.flatten(), "./Figures/RFvals",
                                  filename=f'Coefficients_{model_type}_{target_val}_{usecase}_{thresh_name_full}_{expansion_name}',
                                  fig_title=f"{model_type} Coefficients - {target_val} - Dataset {usecase}", figsize=figsize,
                                  ylabel='Coefficient')



#%% Metrics

    os.makedirs('./Figures/Metrics', exist_ok=True)
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
            plt.bar(x+width*i,vals, width,label=df.iloc[i,0])
        plt.xticks(range(len(df.columns)-1),df.columns[1:])
        plt.grid('both')
        plt.ylabel(metric)
        plt.legend()
        plt.savefig(f'./Figures/Metrics/Metrics_{metric}.png')
        plt.show()

#%% Metrics - adjust file

    timestamp = "20220414_160549"
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
    df.to_csv(os.path.join('./Figures/Metrics',f'Metrics_full_Experiment_{timestamp}_edited.csv'))


#%% feature select

    df = pd.read_csv(os.path.join(result_dir,f'feature_select_full_Experiment_{timestamp}.csv'))
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.drop('Model', axis=0)
    df = df.transpose()
    df_new = pd.DataFrame(index=usecase_names)
    for threshold_set in thresholds_rfvals:
        threshold_name_full = "_".join(name for name in threshold_set)
        selector_name = FeatureSelector.from_name(threshold_set[-1].split("_")[0]).__name__
        for expansion_set in expansion:
            selvals = [df[f'{dataset_name}_{threshold_name_full}_{expansion_set[-1]}_{target_val}_{selector_name}_selected_features'].iloc[0] for dataset_name, target_val in zip(usecase_names, target_vals)]
            allvals = [df[f'{dataset_name}_{threshold_name_full}_{expansion_set[-1]}_{target_val}_{selector_name}_all_features'].iloc[0] for dataset_name, target_val in zip(usecase_names, target_vals)]
            df_new[f'{threshold_name_full}_{expansion_set[-1]}'] = [f'{selval}/{allval}' for selval, allval in zip(selvals, allvals)]
    print(df_new)
    df_new.to_csv('./Figures/featureselect_full.csv',index_label='Threshold')

