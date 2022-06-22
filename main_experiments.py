#%%
import ModelTraining.Preprocessing.add_features as feat_utils
from ModelTraining.feature_engineering.parameters import TrainingParams
from ModelTraining.Utilities import TrainingResults
from ModelTraining.feature_engineering.feature_selectors import FeatureSelectionParams, FeatureSelector
import ModelTraining.Training.TrainingUtilities.training_utils as train_utils
from ModelTraining.Training.run_training_and_test import run_training_model
from ModelTraining.dataimport import DataImport
from ModelTraining.Utilities.MetricsExport import MetricsCalc, ResultExport, metr_utils
import ModelTraining.Preprocessing.data_preprocessing as dp_utils
from ModelTraining.dataimport.data_import import load_from_json
from ModelTraining.feature_engineering.featureset import FeatureSet
from ModelTraining.feature_engineering.expandedmodel import ExpandedModel
from ModelTraining.feature_engineering.feature_expanders import FeatureExpansion
import os
import argparse

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--usecase_names", type=str, default='CPS-Data,SensorA6,SensorB2,SensorC6,Solarhouse1,Solarhouse2')
    parser.add_argument("--model_types", type=str, default='RidgeRegression,LassoRegression,WeightedLS,PLSRegression,RandomForestRegression,RuleFitRegression')
    args = parser.parse_args()
    model_types = model_names = args.model_types.split(",")
    list_usecases = args.usecase_names.split(",")
    data_dir = "../"
    root_dir = "./"
    plot_enabled = False

    # basic training params
    trainparams_basic = TrainingParams.load(os.path.join(root_dir, 'Configuration', 'training_params_normalized.json'))

    # Model parameters and expansion parameters
    parameters_full = {model_type: load_from_json(os.path.join(root_dir, 'Configuration/GridSearchParameters', f'parameters_{model_type}.json')) for model_type in model_types}
    expansion_types = [['IdentityExpander','IdentityExpander'],['IdentityExpander','PolynomialExpansion']]
    expander_parameters = load_from_json(os.path.join(root_dir, 'Configuration','expander_params_PolynomialExpansion.json' ))
    # Feature selection
    list_feature_select_params = [[FeatureSelectionParams('MIC-value',0.05), FeatureSelectionParams('R-value',0.05)]]

    # Use cases
    config_path = os.path.join(root_dir, 'Configuration')
    dict_usecases = [load_from_json(os.path.join(config_path,"UseCaseConfig", f"{name}.json")) for name in
                     list_usecases]

    # Results output
    timestamp = metr_utils.create_file_name_timestamp()
    results_path = os.path.join(root_dir, 'results', timestamp)
    os.makedirs(results_path, exist_ok=True)
    metrics_path = os.path.join(root_dir, 'results', timestamp, 'Metrics')
    os.makedirs(metrics_path, exist_ok=True)

    metrics_names = {'FeatureSelect': ['selected_features', 'all_features'], 'Metrics': ['R2_SKLEARN', 'CV-RMS', 'MAPE', 'RA_SKLEARN'], 'pvalues': ['pvalue_lm', 'pvalue_f']}
    for dict_usecase in dict_usecases:
        for feature_sel_params in list_feature_select_params:
            params_name = "_".join(params.get_full_name() for params in feature_sel_params)
            os.makedirs(os.path.join(results_path, dict_usecase['name'], params_name), exist_ok=True)

#%%
    # Main loop
    print('Starting Training')
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        results_path_dataset = os.path.join(results_path, usecase_name)
        # Get data and feature set
        data_import = DataImport.load(
            os.path.join(config_path, "DataImport", f"{dict_usecase['dataset_filename']}.json"))
        data = data_import.import_data(
            os.path.join(data_dir, dict_usecase['dataset_dir'], dict_usecase['dataset_filename']))
        data = feat_utils.add_features_to_data(data, dict_usecase)
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)
        data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], dict_usecase['dataset_filename'], do_smoothe=False)
        # Main loop
        for feature_sel_params in list_feature_select_params:
            params_name = "_".join(params.get_full_name() for params in feature_sel_params)
            results_path_thresh = os.path.join(results_path_dataset, params_name)
            for expansion in expansion_types:
                for model_type in model_types:
                    list_train_params = [train_utils.set_train_params_model(trainparams_basic, feature_set, feature, model_type, expansion)
                                         for feature in feature_set.get_output_feature_names()]
                    for train_params in list_train_params:
                        model, result = run_training_model(data, train_params,
                                                         feature_select_params=feature_sel_params,
                                                         model_parameters=parameters_full[model_type],
                                                         expander_parameters=expander_parameters,
                                                         prediction_type='ground truth')
                        # Save models
                        model_dir = f"{train_params.model_name}/{train_params.model_type}_{train_params.expansion[-1]}"
                        train_utils.save_model_and_params(model, train_params,
                                                          os.path.join(results_path_thresh, "Models", model_dir))
                        result.save_pkl(results_path_thresh, f'results_{model_type}_{train_params.str_target_feats()}_{train_params.expansion[-1]}.pkl')
    print('Experiments finished')

# %%
    print('Analyzing results')
    metr_exp = MetricsCalc(metr_names=metrics_names)
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)
        for feature_sel_params in list_feature_select_params:
            params_name = "_".join(params.get_full_name() for params in feature_sel_params)
            result_exp = ResultExport(results_root=os.path.join(results_path, usecase_name, params_name),
                                      plot_enabled=True)
            for expansion in expansion_types:
                for model_type in model_types:
                    for feat in feature_set.get_output_feature_names():
                        # Load results
                        result = TrainingResults.load_pkl(result_exp.results_root,
                                                          f'results_{model_type}_{feat}_{expansion[-1]}.pkl')
                        model_dir = os.path.join(result_exp.results_root,
                                                 f"Models/{feat}/{model_type}_{expansion[-1]}/{feat}")
                        model = ExpandedModel.load_pkl(model_dir, "expanded_model.pkl")
                        # Export model properties
                        result_exp.export_model_properties(model)
                        result_exp.export_featsel_metrs(model)
                        result_exp.export_result(result, f"{model_type}_{expansion[-1]}")
                        # Calculate metrics
                        metr_vals_perf = metr_exp.calc_perf_metrics(result, model.get_num_predictors())
                        metr_vals_white = metr_exp.white_test(result)
                        metr_vals_featsel = metr_exp.analyze_featsel(model.transformers.get_transformers_of_type(FeatureSelector))
                        metr_vals = metr_vals_perf + metr_vals_white + metr_vals_featsel
                        # Set metrics identifiers
                        for metr_val in metr_vals_perf:
                            metr_val.set_metr_properties(model_type, model.name, model.transformers.type_last_transf(FeatureExpansion),
                                                         params_name, usecase_name)
                        metr_exp.add_metr_vals(metr_vals_perf)
    metr_exp.store_all_metrics(results_path=metrics_path, timestamp=timestamp)
    print('Result analysis finished')