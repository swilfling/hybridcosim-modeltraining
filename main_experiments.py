import ModelTraining.FeatureEngineering.FeatureCreation.cyclic_features
import ModelTraining.TrainingUtilities.MetricsExport.export_metrics
from ModelTraining.Utilities.Parameters import TrainingParams
from ModelTraining.FeatureEngineering.FeatureSelection import FeatureSelectionParams
import ModelTraining.TrainingUtilities.training_utils as train_utils
from ModelTraining.Training.run_training_and_test import run_training_and_test
import ModelTraining.Utilities.DataPreprocessing.data_preprocessing as dp_utils
import ModelTraining.TrainingUtilities.MetricsExport.export_metrics as metr_exp
import ModelTraining.Utilities.DataProcessing.data_import as data_import
import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--usecase_names", type=str, default='CPS-Data,SensorA6,SensorB2,SensorC6,Solarhouse1,Solarhouse2')
    parser.add_argument("--model_types", type=str, default='RidgeRegression,LassoRegression,WeightedLS,PLSRegression,RandomForestRegression,RuleFitRegression')
    args = parser.parse_args()
    model_types = model_names = args.model_types.split(",")
    list_usecases = args.usecase_names.split(",")
    data_dir = "../"
    root_dir = "./"

    # basic training params
    trainparams_basic = TrainingParams(model_type=model_types[0],
                                       lookback_horizon=4,
                                       prediction_horizon=1,
                                       training_split=0.8,
                                       normalizer="Normalizer",
                                       expansion=['IdentityExpander'])

    # Model parameters and expansion parameters
    parameters_full = {model_type: data_import.load_from_json(os.path.join(root_dir, 'Configuration/GridSearchParameters', f'parameters_{model_type}.json')) for model_type in model_types}
    expansion_types = [['IdentityExpander','IdentityExpander'],['IdentityExpander','PolynomialExpansion']]
    expander_parameters = {'degree': 2, 'interaction_only': True, 'include_bias': False}

    # Use cases
    usecase_config_path = os.path.join(root_dir, 'Configuration/UseCaseConfig')
    solarhouse_usecases = ['Solarhouse1', 'Solarhouse2', 'Solarhouse1_P']
    solarhouse_usecases = []
    inffeld_usecases = ['CPS-Data','SensorA6','SensorB2','SensorC6']
    beyond_usecases = ['Beyond_B12_Gas','Beyond_B20_Gas', 'Beyond_B20_LR']
    #list_usecases = inffeld_usecases + solarhouse_usecases[0:1]

    dict_usecases = [data_import.load_from_json(os.path.join(usecase_config_path, f"{name}.json")) for name in list_usecases]
    plot_enabled = False
    # Feature selection
    list_feature_select_params = [[FeatureSelectionParams('MIC-value',0.05), FeatureSelectionParams('R-value',0.05)]]

    # Results output
    results_path = os.path.join(root_dir, 'results')
    metrics_names = {'FeatureSelect': ['selected_features', 'all_features'], 'Metrics': ['R2_SKLEARN', 'CV-RMS', 'MAPE', 'RA_SKLEARN'], 'pvalues': ['pvalue_lm', 'pvalue_f']}

    # Main loop
    print('Starting Training')
    df_full = pd.DataFrame(index=model_types)

    for dict_usecase in dict_usecases:
        # Get data and feature set
        data, feature_set = data_import.get_data_and_feature_set(os.path.join(data_dir, dict_usecase['dataset']),
                                                                 os.path.join(root_dir, dict_usecase['fmu_interface']))
        usecase_name = dict_usecase['name']

         # Create result directories
        results_path_dataset = os.path.join(results_path, usecase_name)
        os.makedirs(results_path_dataset, exist_ok=True)
        # Export correlation matrices
        metr_exp.export_corrmatrices(data[feature_set.get_input_feature_names()], results_path_dataset,
                                     usecase_name, plot_enabled, expander_parameters=expander_parameters)

        # Add cyclic and statistical features
        ModelTraining.FeatureEngineering.FeatureCreation.cyclic_features.add_cycl_feats(dict_usecase, feature_set)
        data, feature_set = ModelTraining.FeatureEngineering.FeatureCreation.statistical_features.add_stat_feats(data, dict_usecase, feature_set)

        # Main loop
        df_thresh = pd.DataFrame(index=model_types)
        for feature_sel_params in list_feature_select_params:
            params_name = "_".join(params.get_full_name() for params in feature_sel_params)
            results_path_thresh = os.path.join(results_path_dataset, params_name)
            os.makedirs(results_path_thresh, exist_ok=True)
            for expansion in expansion_types:
                df_metrics_models = pd.DataFrame()
                for model_type in model_types:
                    smoothe_data = True if model_type != 'RandomForestRegression' else False
                    data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], smoothe_data=smoothe_data)
                    list_training_parameters = [train_utils.set_train_params_model(trainparams_basic, feature_set, feature, model_type, expansion)
                                                for feature in feature_set.get_output_feature_names()]
                    models, [results, df_metrics] = run_training_and_test(data, list_training_parameters, results_path_thresh,
                                                                    do_predict=True, feature_select_params=feature_sel_params,
                                                                    model_parameters=parameters_full[model_type], expander_parameters=expander_parameters, plot_enabled=plot_enabled,
                                                                    metrics_names=metrics_names, prediction_type='ground truth')
                    df_metrics_models = df_metrics_models.append(df_metrics)
                df_thresh = df_thresh.join(df_metrics_models.add_prefix(f'{params_name}_{expansion[-1]}_'))
            df_thresh.to_csv(os.path.join(results_path_thresh, f'Metrics_{usecase_name}_{params_name}.csv'))

        df_full = df_full.join(df_thresh.add_prefix(f'{usecase_name}_'))

    ModelTraining.TrainingUtilities.MetricsExport.export_metrics.store_all_metrics(df_full, results_path=results_path, metrics_names=metrics_names)
    print('Experiments finished')