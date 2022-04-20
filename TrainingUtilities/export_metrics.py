import os
import pandas as pd

from ModelTraining.datamodels.datamodels.validation import metrics as metrics
from ModelTraining.Utilities.DataProcessing import data_export as data_export
from ModelTraining.Utilities.Plotting import plotting_utilities as plt_utils
import ModelTraining.TrainingUtilities.training_utils as train_utils


class MetricsExport:
    plot_enabled: bool = False
    results_root: str = "./"

    def __init__(self, plot_enabled=False, results_root="./", metr_dir='Metrics'):
        self.plot_enabled = plot_enabled
        self.results_root = results_root

    def export_rvals(self, expanders, selectors, feature_names):
        results_dir_path = os.path.join(self.results_root, 'FeatureSelection')
        os.makedirs(results_dir_path, exist_ok=True)
        for expander, selector in zip(expanders, selectors):
            feature_names = expander.get_feature_names(feature_names)
            rvals = selector.coef_[selector.get_support()]
            df = pd.DataFrame(data=rvals, index=feature_names)
            df.to_csv(os.path.join(results_dir_path, f'{selector.__class__.__name__} - {expander.__class__.__name__}.csv'), index_label='Feature')
            if self.plot_enabled:
                plt_utils.barplot(df.columns, df.values.flatten(), fig_save_path=results_dir_path,
                                  fig_title=f'{selector.__class__.__name__} - {expander.__class__.__name__}',
                                  ylabel=str(selector.__class__.__name__), figsize=(30, 7))

    def export_model_properties(self, model, metrics_dir):
        feature_names = model.get_expanded_feature_names()
        model_types_coef = ['RandomForestRegression','RidgeRegression', 'LinearRegression']
        title = f'{model.__class__.__name__}_{model.expanders[-1].__class__.__name__}'
        # Export coefficients
        if model.__class__.__name__ in model_types_coef:
            if model.__class__.__name__ == 'RandomForestRegression':
                #        graphviz_utils.visualize_rf(model, feature_names, outfile_name=os.path.join(metrics_dir,f"rf_{title}"), depth=4)
                df = pd.DataFrame(data=[model.model.feature_importances_.tolist()], columns=feature_names)
            # Ridge
            else:
                df = pd.DataFrame(data=[model.model.coef_[0]], columns=feature_names)
            ylabel = 'F-Score' if model.__class__.__name__ == 'RandomForestRegression' else 'Coefficients'
            if self.plot_enabled:
                plt_utils.barplot(df.columns, df.values.flatten(), metrics_dir, f'Coefficients_{title}', ylabel=ylabel, figsize=(30, 7))
            df.to_csv(os.path.join(metrics_dir, f'Coefficients_{title}.csv'))
        else:
            if model.__class__.__name__ == 'SymbolicRegression':
                data_export.metrics_to_json({'Program': str(model.model._program)}, os.path.join(metrics_dir, f'Program_{title}.json'))
            if model.__class__.__name__ == 'RuleFitRegression':
                rules = model.model.get_rules()
                rules = rules[rules.coef != 0].sort_values("support", ascending=False)
                rules.to_csv(os.path.join(metrics_dir, f'Rules_{title}.csv'), float_format="%.2f")

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

    def export_results(self, model, target_features, result_df):
        result_dir_names = ['Plots','ModelProperties']
        [plot_dir, metrics_dir] = [os.path.join(self.results_root, name) for name in result_dir_names]
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(metrics_dir, exist_ok=True)
        title = f'{model.__class__.__name__}_{model.expanders[-1].__class__.__name__}'
        # Export results
        self.export_timeseries_results(result_df, target_features, plot_dir, title)
        self.export_model_properties(model, metrics_dir)


def update_metr_df(df_full, dict_data, prefix="", suffix=""):
    df_new = pd.DataFrame(data=dict_data, index=df_full.index)
    return df_full.join(df_new.add_suffix(suffix).add_prefix(prefix))


def export_corrmatrices(data, metrics_path, filename, plot_enabled=True, expander_parameters={}):
    if data.shape[1] > 1:
        plt_utils.printHeatMap(data, metrics_path,
                           f'Correlation_{filename}_IdentityExpander',
                           plot_enabled=plot_enabled)
        expanded_features = train_utils.expand_features(data, data.columns, [], expander_parameters=expander_parameters)
        plt_utils.printHeatMap(expanded_features, metrics_path,
                           f'Correlation_{filename}_PolynomialExpansion',
                           plot_enabled=plot_enabled)


def calc_metrics(y_true,y_pred, no_samples=0, n_predictors=0,metrics_names=['R2','CV-RMS', 'MAPE']):
    metrs_feature = metrics.all_metrics(y_true=y_true, y_pred=y_pred)
    metrs_feature = {k:v for k,v in metrs_feature.items() if k in metrics_names}
    if 'RA' in metrics_names:
        metrs_feature.update({"RA": metrics.rsquared_adj(y_true, y_pred, no_samples, n_predictors)})
    if 'RA_SKLEARN' in metrics_names:
        metrs_feature.update({"RA_SKLEARN": metrics.rsquared_sklearn_adj(y_true, y_pred, no_samples, n_predictors)})
    # Select metrics
    return metrs_feature
