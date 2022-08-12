#%%
import pandas as pd
import os

from ModelTraining.Data.Plotting import plot_data
from ModelTraining.feature_engineering.featureengineering.featureselectors import FeatureSelector


if __name__ == "__main__":
    # %%
    usecase_names = ['CPS-Data','SensorA6', 'SensorB2', 'SensorC6', 'Solarhouse1', 'Solarhouse2']
    target_vals = ['energy'] * 4 + ['TSolarVL','T_Solar_VL']
    solarhouse_usecases = ['Solarhouse1','Solarhouse2']

    thresholds_rfvals = [['MIC-value_0.05','R-value_0.05']]
    expansion = [['IdentityExpander','IdentityExpander'], ['IdentityExpander','PolynomialExpansion']]

    model_types = ['RidgeRegression']
    timestamp = '20220509_113409'
    experiment_name = f'Experiment_{timestamp}'
    result_dir = os.path.join('../', 'results', experiment_name)
    output_dir = f"./Figures/{experiment_name}"

    fvals_dir = os.path.join(output_dir, 'FeatureSelection')
    os.makedirs(fvals_dir, exist_ok=True)

    #%% Fvals
    for usecase in usecase_names:
        for threshold_set, expansion_set in zip(thresholds_rfvals, expansion):
            thresh_name_full = "_".join(name for name in threshold_set)
            path = os.path.join(result_dir, usecase, thresh_name_full, 'FeatureSelection')
            for threshold, expansion_name in zip(threshold_set, expansion_set):
                threshold_name = threshold.split("_")[0]
                selector_name = FeatureSelector.cls_from_name(threshold_name).__name__
                df = pd.read_csv(os.path.join(path, f'Coefficients_{selector_name} - {expansion_name}.csv'), index_col='Feature')
                if usecase in solarhouse_usecases:
                    figsize = (5,5) if expansion_name == 'IdentityExpander' else (6,6)
                else:
                    figsize = (8,4) if expansion_name == 'IdentityExpander' else (20,15)
                filename = f'{threshold}_{usecase}_{thresh_name_full}_{expansion}'
                fig_title = f"{threshold_name} - Dataset {usecase}"
                plot_data.barplot(df, fvals_dir, filename=filename, fig_title=fig_title, figsize=figsize,
                                           ylabel=threshold_name)


#%% feature select
    df = pd.read_csv(os.path.join(result_dir,f'feature_select_full_Experiment_{timestamp}.csv'))
    df = df.transpose()
    df.columns = df.iloc[0]
    df = df.drop('Model', axis=0)
    df = df.transpose()
    df_new = pd.DataFrame(index=usecase_names)
    for threshold_set in thresholds_rfvals:
        threshold_name_full = "_".join(name for name in threshold_set)
        selector_name = FeatureSelector.cls_from_name(threshold_set[-1].split("_")[0]).__name__
        for expansion_set in expansion:
            selvals = [df[f'{dataset_name}_{threshold_name_full}_{expansion_set[-1]}_{target_val}_{selector_name}_selected_features'].iloc[0] for dataset_name, target_val in zip(usecase_names, target_vals)]
            allvals = [df[f'{dataset_name}_{threshold_name_full}_{expansion_set[-1]}_{target_val}_{selector_name}_all_features'].iloc[0] for dataset_name, target_val in zip(usecase_names, target_vals)]
            df_new[f'{threshold_name_full}_{expansion_set[-1]}'] = [f'{selval}/{allval}' for selval, allval in zip(selvals, allvals)]
    print(df_new)
    df_new.to_csv(os.path.join(fvals_dir,'featureselect_full.csv'),index_label='Threshold')

