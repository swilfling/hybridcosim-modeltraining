import ModelTraining.Preprocessing.FeatureCreation.add_features as feat_utils
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.Preprocessing.FeatureSelection import FeatureSelectionParams
from ModelTraining.Preprocessing.FeatureSelection.feature_selectors import FeatureSelector
import ModelTraining.Training.TrainingUtilities.training_utils as train_utils
from ModelTraining.Utilities.MetricsExport.MetricsCalc import MetricsCalc
from ModelTraining.Utilities.MetricsExport.ResultExport import ResultExport
import ModelTraining.Preprocessing.DataImport.data_import as data_import
from ModelTraining.Preprocessing.feature_set import FeatureSet
from ModelTraining.datamodels.datamodels import Model
import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--usecase_names", type=str, default='CPS-Data,SensorA6,SensorB2,SensorC6,Solarhouse1,Solarhouse2')
    parser.add_argument("--model_types", type=str, default='RidgeRegression,LassoRegression,RandomForestRegression')
    args = parser.parse_args()
    model_types = model_names = args.model_types.split(",")
    list_usecases = args.usecase_names.split(",")
    data_dir = "../"
    root_dir = "./"
    plot_enabled = False

    # basic training params
    trainparams_basic = TrainingParams.load(os.path.join(root_dir, 'Configuration', 'training_params_normalized.json'))

    # Model parameters and expansion parameters
    parameters_full = {model_type: data_import.load_from_json(os.path.join(root_dir, 'Configuration/GridSearchParameters', f'parameters_{model_type}.json')) for model_type in model_types}
    expansion_types = [['IdentityExpander','IdentityExpander'],['IdentityExpander','PolynomialExpansion']]
    expander_parameters = data_import.load_from_json(os.path.join(root_dir, 'Configuration','expander_params_PolynomialExpansion.json' ))
    # Feature selection
    list_feature_select_params = [[FeatureSelectionParams('MIC-value',0.05), FeatureSelectionParams('R-value',0.05)]]

    # Use cases
    usecase_config_path = os.path.join(root_dir, 'Configuration/UseCaseConfig')
    dict_usecases = [data_import.load_from_json(os.path.join(usecase_config_path, f"{name}.json")) for name in
                     list_usecases]

    # Results output
    timestamp = "Experiment_20220609_113050"
    results_path = os.path.join(root_dir, 'results', timestamp)
    os.makedirs(results_path, exist_ok=True)
    metrics_path = os.path.join(root_dir, 'results', timestamp, 'Metrics')
    os.makedirs(metrics_path, exist_ok=True)
    metrics_names = {'FeatureSelect': ['selected_features', 'all_features'], 'Metrics': ['R2_SKLEARN', 'CV-RMS', 'MAPE', 'RA_SKLEARN'], 'pvalues': ['pvalue_lm', 'pvalue_f']}


    # %%
    print('Analyzing results')
    df_full = pd.DataFrame()
    metr_exp = MetricsCalc(metr_names=metrics_names)
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        results_path_dataset = os.path.join(results_path, usecase_name)
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(dict_usecase, feature_set)
        # Main loop
        df_thresh = pd.DataFrame(index=model_types)
        for feature_sel_params in list_feature_select_params:
            params_name = "_".join(params.get_full_name() for params in feature_sel_params)
            res_dir_thresh = os.path.join(results_path_dataset, params_name)
            result_exp = ResultExport(results_root=res_dir_thresh, plot_enabled=False)
            for expansion in expansion_types:
                df_metrics_models = pd.DataFrame()
                for model_type in model_types:
                    df_metrics = pd.DataFrame()
                    for feature in feature_set.get_output_feature_names():
                        train_params = train_utils.set_train_params_model(trainparams_basic, feature_set, feature,
                                                                          model_type, expansion)
                        result = TrainingResults.load_pkl(res_dir_thresh,
                                                          f'results_{model_type}_{train_params.str_target_feats()}_{train_params.expansion[-1]}.pkl')
                        model = Model.load(os.path.join(res_dir_thresh,
                                                        f"Models/{train_params.model_name}/{train_params.model_type}_{train_params.expansion[-1]}/{feature}"))
                        selectors = [FeatureSelector.load_pkl(res_dir_thresh,
                                                              f'FeatureSelection/{train_params.model_name}/{train_params.model_type}_{train_params.expansion[-1]}/selector_{i}.pkl')
                                     for i, _ in enumerate(train_params.expansion)]
                        result_exp.export_model_properties(model)
                        model_name = f'{model.name}_{model.expanders.type_last_exp()}'
                        model_name_full = f'{model.__class__.__name__}_{model_name}'
                        result_exp.export_result(result, model_name_full)
                        df_met = metr_exp.analyze_result(model, result).join(metr_exp.analyze_featsel(model, selectors))
                        df_metrics = df_met if df_metrics.empty else df_metrics.join(df_met)
                    df_metrics_models = df_metrics_models.append(df_metrics)
                df_thresh = df_thresh.join(df_metrics_models.add_prefix(f'{params_name}_'))
            df_thresh.to_csv(os.path.join(res_dir_thresh, f'Metrics_{usecase_name}_{params_name}.csv'))
        df_full = df_thresh.add_prefix(f'{usecase_name}_') if df_full.empty else df_full.join(
            df_thresh.add_prefix(f'{usecase_name}_'))
    metr_exp.store_all_metrics(df_full, results_path=metrics_path, timestamp=timestamp)
    print('Result analysis finished')