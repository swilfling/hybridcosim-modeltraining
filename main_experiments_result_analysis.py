import ModelTraining.Preprocessing.FeatureCreation.add_features as feat_utils
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.Preprocessing.FeatureSelection import FeatureSelectionParams
from ModelTraining.Preprocessing.FeatureSelection.feature_selectors import FeatureSelector
from ModelTraining.Utilities.MetricsExport.metrics_calc import MetricsCalc
from ModelTraining.Utilities.MetricsExport.result_export import ResultExport
import ModelTraining.Preprocessing.DataImport.data_import as data_import
from ModelTraining.Preprocessing.featureset import FeatureSet
from ModelTraining.datamodels.datamodels.wrappers.feature_extension import ExpandedModel
import os
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
    timestamp = "Experiment_20220610_121221"
    results_path = os.path.join(root_dir, 'results', timestamp)
    os.makedirs(results_path, exist_ok=True)
    metrics_path = os.path.join(root_dir, 'results', timestamp, 'Metrics')
    os.makedirs(metrics_path, exist_ok=True)
    metrics_names = {'FeatureSelect': ['selected_features', 'all_features'], 'Metrics': ['R2_SKLEARN', 'CV-RMS', 'MAPE', 'RA_SKLEARN'], 'pvalues': ['pvalue_lm', 'pvalue_f']}

    # %%
    print('Analyzing results')
    metr_exp = MetricsCalc(metr_names=metrics_names)
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(dict_usecase, feature_set)
        for feature_sel_params in list_feature_select_params:
            params_name = "_".join(params.get_full_name() for params in feature_sel_params)
            result_exp = ResultExport(results_root=os.path.join(results_path, usecase_name, params_name), plot_enabled=True)
            for expansion in expansion_types:
                for model_type in model_types:
                    for feat in feature_set.get_output_feature_names():
                        # Load results
                        result = TrainingResults.load_pkl(result_exp.results_root, f'results_{model_type}_{feat}_{expansion[-1]}.pkl')
                        model = ExpandedModel.load_pkl(os.path.join(result_exp.results_root, f"Models/{feat}/{model_type}_{expansion[-1]}/{feat}"), "expanded_model.pkl")
                        #model = Model.load(os.path.join(result_exp.results_root, f"Models/{feat}/{model_type}_{expansion[-1]}/{feat}"))
                        selectors = [FeatureSelector.load_pkl(result_exp.results_root, f'FeatureSelection/{feat}/{model_type}_{expansion[-1]}/selector_{i}.pkl')
                                     for i, _ in enumerate(expansion)]
                        # Export model properties
                        result_exp.export_model_properties(model)
                        result_exp.export_result(result, f"{model_type}_{expansion[-1]}")
                        # Calculate metrics
                        metr_vals_perf = metr_exp.calc_perf_metrics(result, model.expanders.get_num_output_feats())
                        metr_vals_white = metr_exp.white_test(result)
                        metr_vals_featsel = metr_exp.analyze_featsel(selectors)
                        metr_vals = metr_vals_perf + metr_vals_white + metr_vals_featsel
                        # Set metrics identifiers
                        for metr_val in metr_vals_perf:
                            metr_val.set_metr_properties(model_type, model.name, model.expanders.type_last_exp(),
                                                         params_name, usecase_name)
                        metr_exp.add_metr_vals(metr_vals_perf)
    metr_exp.store_all_metrics(results_path=metrics_path, timestamp=timestamp)
    print('Result analysis finished')