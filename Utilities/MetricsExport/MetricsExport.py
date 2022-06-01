import os

import pandas as pd
from ModelTraining.datamodels.datamodels.validation.white_test import white_test
from .export_metrics import update_metr_df, calc_metrics, metrics_to_json
from ModelTraining.Utilities.Plotting import plotting_utilities as plt_utils


class MetricsExport:
    plot_enabled: bool = False
    results_root: str = "./"
    result_dir = 'Results'
    scatter_dir = 'Scatter'
    tseries_dir = 'Timeseries'

    def __init__(self, plot_enabled=False, results_root="./", metr_dir='Metrics'):
        self.plot_enabled = plot_enabled
        self.results_root = results_root

    def create_dirs(self):
        # Create dirs
        plot_dir = os.path.join(self.results_root, self.result_dir)
        os.makedirs(plot_dir, exist_ok=True)
        if self.plot_enabled:
            scatter_plot_dir = os.path.join(plot_dir, self.scatter_dir)
            ts_plot_dir = os.path.join(plot_dir, self.tseries_dir)
            os.makedirs(scatter_plot_dir, exist_ok=True)
            os.makedirs(ts_plot_dir, exist_ok=True)

    def get_scatter_dir(self):
        return os.path.join(self.results_root, self.result_dir, self.scatter_dir)

    def get_tseries_dir(self):
        return os.path.join(self.results_root, self.result_dir, self.tseries_dir)

    def get_result_dir(self):
        return os.path.join(self.results_root, self.result_dir)

    def export_coeffs(self, coeffs, feature_names, dir, title, ylabel=""):
        df = pd.DataFrame(data=coeffs, index=feature_names)
        df.to_csv(os.path.join(dir, f'Coefficients_{title}.csv'), float_format='%.2f', index_label='Feature')
        if self.plot_enabled:
            for col in df.columns:
                plt_utils.barplot(df.index, df[col].to_numpy(), dir, f'Coefficients_{title}', ylabel=ylabel,
                                  figsize=(30, 7))

    def export_featsel_metrs(self, expanders, selectors, feature_names):
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

    def export_model_properties(self, model):
        property_dir = os.path.join(self.results_root, 'ModelProperties')
        os.makedirs(property_dir, exist_ok=True)
        model_type = model.__class__.__name__
        title = f'{model_type}_{model.name}_{model.expanders[-2].__class__.__name__}'
        # Export coefficients
        if model_type in ['RandomForestRegression', 'RidgeRegression', 'LinearRegression']:
            ylabel = 'F-Score' if model_type == 'RandomForestRegression' else 'Coefficients'
            self.export_coeffs(model.get_coef().T, model.get_expanded_feature_names(), property_dir, title, ylabel)
        if model_type == 'SymbolicRegression':
            metrics_to_json({'Program': str(model.get_program())}, os.path.join(property_dir, f'Program_{title}.json'))
        if model_type == 'RuleFitRegression':
            model.get_rules().to_csv(os.path.join(property_dir, f'Rules_{title}.csv'), float_format="%.2f", index_label='Rule')


def analyze_result(models, results, list_training_parameters, list_selectors=[], plot_enabled=True, results_dir_path="", **kwargs):
    metrics_exp = MetricsExport(plot_enabled=plot_enabled, results_root=results_dir_path)
    df_metr_full = pd.DataFrame(index=[models[0].__class__.__name__])
    metr_names = kwargs.get('metrics_names', {'Metrics': ['R2', 'CV-RMS', 'MAPE'],
                                                 'FeatureSelect': ['selected_features', 'all_features'],
                                                 'pvalues': ['pvalues_lm']})
    # Export feature selection metrics
    for model, selectors in zip(models,list_selectors):
        metrics_exp.export_featsel_metrs(model.expanders, selectors, model.feature_names)
        for selector in selectors:
            df_metr_full = update_metr_df(df_metr_full, selector.get_metrics(),
                                                      prefix=f'{model.name}_{selector.__class__.__name__}_',
                                                      suffix='_FeatureSelect')
    metrics_exp.create_dirs()
    for model, result, train_params in zip(models, results, list_training_parameters):
        model_type = model.__class__.__name__
        final_expander_type = model.expanders[-2].__class__.__name__
        title = f'{model_type}_{model.name}_{final_expander_type}'
        num_samples_total = result.train_index.shape[0] + result.test_index.shape[0]
        num_predictors = len(model.get_expanded_feature_names())
        result.test_results_to_csv(metrics_exp.get_result_dir(),f'Timeseries_{title}.csv')

        for feat in result.target_feat_names:
            # Store results for feature in csv
            y_true = result.test_target_vals(feat)
            y_pred = result.test_pred_vals(feat)
            # Calculate metrics
            metrs = calc_metrics(y_true, y_pred, num_samples_total, num_predictors, metrics_names=metr_names['Metrics'])
            df_metr_full = update_metr_df(df_metr_full, metrs, prefix=f'{model.name}_{feat}_', suffix='_Metrics')
            # White test
            white_pvals = white_test(result.test_input, y_true - y_pred)
            df_metr_full = update_metr_df(df_metr_full, white_pvals, prefix=f'{model.name}_{feat}_', suffix='_pvalues')
            # Plot results
            if metrics_exp.plot_enabled:
                plt_utils.plot_result(result.test_result_df(feat), metrics_exp.get_tseries_dir(), f"Timeseries_{feat}_{title}", store_to_csv=False, figsize=(14, 4))
                plt_utils.scatterplot(y_pred, y_true, metrics_exp.get_scatter_dir(), f"Scatter_{feat}_{title}")

        # Export Model Properties
        metrics_exp.export_model_properties(model)
        
    return df_metr_full