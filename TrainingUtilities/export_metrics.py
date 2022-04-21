import pandas as pd

from ModelTraining.datamodels.datamodels.validation import metrics as metrics
from ModelTraining.Utilities.Plotting import plotting_utilities as plt_utils
import ModelTraining.TrainingUtilities.training_utils as train_utils


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
