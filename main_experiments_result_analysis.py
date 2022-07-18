import ModelTraining.Preprocessing.add_features as feat_utils
from ModelTraining.feature_engineering.parameters import TrainingParams, TrainingParamsExpanded, TransformerParams
from ModelTraining.Utilities import TrainingData
from ModelTraining.feature_engineering.feature_selectors import FeatureSelector
from ModelTraining.feature_engineering.feature_expanders import FeatureExpansion
from ModelTraining.Utilities.MetricsExport.metrics_calc import MetricsCalc
from ModelTraining.Utilities.MetricsExport.result_export import ResultExport
from ModelTraining.dataimport.data_import import load_from_json
from ModelTraining.feature_engineering.featureset import FeatureSet
from ModelTraining.feature_engineering.expandedmodel import ExpandedModel
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--usecase_names", type=str, default='CPS-Data,SensorA6,SensorB2,SensorC6,Solarhouse1,Solarhouse2')
    parser.add_argument("--model_types", type=str, default='RidgeRegression,LassoRegression,WeightedLS,PLSRegression,RandomForestRegression,RuleFitRegression')
    args = parser.parse_args()
    model_types = model_names = args.model_types.split(",")
    list_usecases = args.usecase_names.split(",")
    data_dir = "../"
    root_dir = "./"
    plot_enabled = False

    # Model parameters and expansion parameters
    parameters_full = {model_type: load_from_json(os.path.join(root_dir, 'Configuration/GridSearchParameters', f'parameters_{model_type}.json')) for model_type in model_types}
    expander_parameters = load_from_json(os.path.join(root_dir, 'Configuration','expander_params_PolynomialExpansion.json' ))

    transf_cfg_files = [f"train_params_mic_0_05_{expansion_type}_r_0_05.json" for expansion_type in
                        ['basic', 'poly']]
    list_train_params = [
        TrainingParamsExpanded.load(os.path.join(root_dir, "Configuration", "TrainingParameters", file)) for file in
        transf_cfg_files]
    params_names = ['MIC-value_0.05_R-value_0.05']

    # Use cases
    config_path = os.path.join(root_dir, 'Configuration')
    dict_usecases = [load_from_json(os.path.join(config_path,"UseCaseConfig", f"{name}.json")) for name in
                     list_usecases]


    # Results output
    timestamp = "Experiment_20220615_120631"
    results_path = os.path.join(root_dir, 'results', timestamp)
    os.makedirs(results_path, exist_ok=True)
    metrics_path = os.path.join(root_dir, 'results', timestamp, 'Metrics')
    os.makedirs(metrics_path, exist_ok=True)
    metrics_names = {'FeatureSelect': ['selected_features', 'all_features'], 'Metrics': ['rsquared', 'cvrmse', 'mape', 'RA'], 'pvalues': ['pvalue_lm', 'pvalue_f']}

# %%
    print('Analyzing results')
    metr_exp = MetricsCalc(metr_names=metrics_names)
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)
        for params_name in params_names:
            result_exp = ResultExport(results_root=os.path.join(results_path, usecase_name, params_name),
                                      plot_enabled=True)
            for training_params in list_train_params:
                for model_type in model_types:
                    for feat in feature_set.get_output_feature_names():
                        # Load results
                        result = TrainingData.load_pkl(result_exp.results_root,
                                                          f'results_{model_type}_{feat}_{training_params.str_expansion()}.pkl') # TODO fix this
                        model_dir = os.path.join(result_exp.results_root,
                                                 f'Models/{feat}/{model_type}_{training_params.str_expansion()}/{feat}')
                        model = ExpandedModel.load_pkl(model_dir, "expanded_model.pkl")
                        result_exp.export_result_full(model, result, model.transformers.type_transf_full())
                        # Calculate metrics
                        selectors = model.transformers.get_transformers_of_type(FeatureSelector)
                        metr_vals = metr_exp.calc_all_metrics(result, selectors, model.get_num_predictors())
                        # Set metrics identifiers
                        for metr_val in metr_vals:
                            metr_val.set_metr_properties(model_type, model.name,
                                                         model.transformers.type_last_transf(FeatureExpansion),
                                                         params_name, usecase_name)
                        metr_exp.add_metr_vals(metr_vals)
    metr_exp.store_all_metrics(results_path=metrics_path, timestamp=timestamp)
    print('Result analysis finished')