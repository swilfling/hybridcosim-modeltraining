import os

import pandas as pd
import numpy as np
import ModelTraining.datamodels
from .export_metrics import update_metr_df, calc_metrics, metrics_to_json
from ModelTraining.Utilities.Plotting import plotting_utilities as plt_utils


class MetricsExport:
    plot_enabled: bool = False
    results_root: str = "./"

    def __init__(self, plot_enabled=False, results_root="./", metr_dir='Metrics'):
        self.plot_enabled = plot_enabled
        self.results_root = results_root

    def export_coeffs(self, coeffs, feature_names, dir, title, ylabel=""):
        df = pd.DataFrame(data=coeffs, index=feature_names)
        df.to_csv(os.path.join(dir, f'Coefficients_{title}.csv'), float_format='%.2f', index_label='Feature')
        if self.plot_enabled:
            for col in df.columns:
                plt_utils.barplot(df.index, df[col].values, dir, f'Coefficients_{title}', ylabel=ylabel,
                                  figsize=(30, 7))

    def export_rvals(self, expanders, selectors, feature_names):
        results_dir_path = os.path.join(self.results_root, 'FeatureSelection')
        os.makedirs(results_dir_path, exist_ok=True)
        for i, selector in enumerate(selectors):
            selector.save_pkl(results_dir_path, f"{selector.__class__.__name__}_{i}.pickle")
        for expander, selector in zip(expanders, selectors):
            feature_names = expander.get_feature_names_out(feature_names)
            self.export_coeffs(selector.get_coef(),
                               feature_names=feature_names,
                               dir=results_dir_path,
                               title=f'{selector.__class__.__name__} - {expander.__class__.__name__}',
                               ylabel=str(selector.__class__.__name__))

    def export_model_properties(self, model, metrics_dir):
        model_type = model.__class__.__name__
        title = f'{model_type}_{model.name}_{model.expanders[-2].__class__.__name__}'
        # Export coefficients
        if model_type in ['RandomForestRegression', 'RidgeRegression', 'LinearRegression']:
            ylabel = 'F-Score' if model_type == 'RandomForestRegression' else 'Coefficients'
            self.export_coeffs(model.get_coef().T, model.get_expanded_feature_names(), metrics_dir, title, ylabel)
        if model_type == 'SymbolicRegression':
            metrics_to_json({'Program': str(model.get_program())}, os.path.join(metrics_dir, f'Program_{title}.json'))
        if model_type == 'RuleFitRegression':
            model.get_rules().to_csv(os.path.join(metrics_dir, f'Rules_{title}.csv'), float_format="%.2f", index_label='Rule')

    def export_timeseries_results(self, result_df, target_features, plot_dir, title):
        for feature in target_features:
            result_df[[feature, f'predicted_{feature}']].to_csv(os.path.join(plot_dir, f'Timeseries_{title}.csv'))
        if self.plot_enabled:
            scatter_plot_dir = os.path.join(plot_dir, 'Scatter')
            ts_plot_dir = os.path.join(plot_dir, 'Timeseries')
            os.makedirs(scatter_plot_dir, exist_ok=True)
            os.makedirs(ts_plot_dir, exist_ok=True)
            for feature in target_features:
                unit = "°C" if feature == 'TSolarVL' or feature == 'T_Solar_VL' or feature == 'TB20BR1' else "kWh"
                ylabel = f"{feature} [{unit}], predicted_{feature} [{unit}]"
                plt_utils.plot_result(result_df[[feature, f'predicted_{feature}']], ts_plot_dir, f"Timeseries_{title}",
                                      store_to_csv=False, ylabel=ylabel, figsize=(14, 4))
                plt_utils.scatterplot(result_df[f'predicted_{feature}'], result_df[feature], scatter_plot_dir, f"Scatter_{title}")

    def export_results(self, model, target_features, test_index, test_target, test_prediction):
        result_dir_names = ['Plots','ModelProperties']
        [plot_dir, metrics_dir] = [os.path.join(self.results_root, name) for name in result_dir_names]
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        # Export results
        title = f'{model.__class__.__name__}_{model.name}_{model.expanders[-2].__class__.__name__}'
        result_df = pd.DataFrame(np.hstack((test_target, test_prediction)),index=test_index, columns=target_features + [f'predicted_{feature}' for feature in target_features])
        self.export_timeseries_results(result_df, target_features, plot_dir, title)
        self.export_model_properties(model, metrics_dir)


def analyze_result(models, results, list_training_parameters, list_selectors=[], plot_enabled=True, results_dir_path="", **kwargs):
    metrics_exp = MetricsExport(plot_enabled=plot_enabled, results_root=results_dir_path)
    df_metrics_full = pd.DataFrame(index=[models[0].__class__.__name__])
    metrics_names = kwargs.get('metrics_names', {'Metrics': ['R2', 'CV-RMS', 'MAPE'],
                                                 'FeatureSelect': ['selected_features', 'all_features'],
                                                 'pvalues': ['pvalues_lm']})
    # Export feature selection metrics
    for model, selectors in zip(models,list_selectors):
        metrics_exp.export_rvals(model.expanders, selectors, model.feature_names)
        for selector in selectors:
            df_metrics_full = update_metr_df(df_metrics_full, selector.get_metrics(),
                                                      prefix=f'{model.name}_{selector.__class__.__name__}_',
                                                      suffix='_FeatureSelect')

    for model, result, train_params in zip(models, results, list_training_parameters):
        metrics_exp.export_results(model, train_params.target_features, result.test_index, result.test_target, result.test_prediction)
        for i, feature in enumerate(train_params.target_features):
            y_true = result.test_target[:,i]
            y_pred = result.test_prediction[:,i]
            # Calculate Metrics
            metrics = calc_metrics(y_true,y_pred, result.train_index.shape[0] + result.test_index.shape[0], len(model.get_expanded_feature_names()),
                                            metrics_names=metrics_names['Metrics'])
            df_metrics_full = update_metr_df(df_metrics_full, metrics,
                                                      prefix=f'{model.name}_{feature}_', suffix='_Metrics')
            # White test
            white_test_results = ModelTraining.datamodels.datamodels.validation.white_test.white_test(result.test_input, y_true - y_pred)
            df_metrics_full = update_metr_df(df_metrics_full, white_test_results,
                                                      prefix=f'{model.name}_{feature}_', suffix='_pvalues')

    return df_metrics_full