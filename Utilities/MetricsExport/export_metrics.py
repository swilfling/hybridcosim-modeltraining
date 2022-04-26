import json
import os
from datetime import datetime

import pandas as pd

from ModelTraining.datamodels.datamodels.validation import metrics as metrics
import ModelTraining.Preprocessing.FeatureCreation.cyclic_features as cyc_feats
from ModelTraining.Utilities.Plotting import plotting_utilities as plt_utils
import ModelTraining.Training.TrainingUtilities.training_utils as train_utils


def update_metr_df(df_full, dict_data, prefix="", suffix=""):
    df_new = pd.DataFrame(data=dict_data, index=df_full.index)
    return df_full.join(df_new.add_suffix(suffix).add_prefix(prefix))


def export_corrmatrices(data, metrics_path, filename, plot_enabled=True, expander_parameters={}):
    features_to_exclude = cyc_feats.get_cyc_feature_names()
    data = data[[col for col in data.columns if col not in features_to_exclude]]
    if data.shape[1] > 1:
        plt_utils.printHeatMap(data, metrics_path,
                           f'Correlation_{filename}_IdentityExpander',
                           plot_enabled=plot_enabled)
        expanded_features = train_utils.expand_features(data, data.columns, [], expander_parameters=expander_parameters)
        plt_utils.printHeatMap(expanded_features, metrics_path,
                           f'Correlation_{filename}_PolynomialExpansion',
                           plot_enabled=plot_enabled, annot=False)


def calc_metrics(y_true,y_pred, no_samples=0, n_predictors=0,metrics_names=['R2','CV-RMS', 'MAPE']):
    metrs_feature = metrics.all_metrics(y_true=y_true, y_pred=y_pred)
    metrs_feature = {k:v for k,v in metrs_feature.items() if k in metrics_names}
    if 'RA' in metrics_names:
        metrs_feature.update({"RA": metrics.rsquared_adj(y_true, y_pred, no_samples, n_predictors)})
    if 'RA_SKLEARN' in metrics_names:
        metrs_feature.update({"RA_SKLEARN": metrics.rsquared_sklearn_adj(y_true, y_pred, no_samples, n_predictors)})
    # Select metrics
    return metrs_feature


def store_all_metrics(df_full, results_path, metrics_names):
    timestamp = create_file_name_timestamp()
    df_full.to_csv(os.path.join(results_path, f'AllMetrics_full_{timestamp}.csv'),index_label='Model', float_format='%.3f')

    df_metrics = get_df_subset(df_full, 'Metrics')
    df_featsel = get_df_subset(df_full, 'FeatureSelect')
    df_whitetest = get_df_subset(df_full, 'pvalues')

    for df, filename in zip([df_metrics, df_featsel, df_whitetest], ['Metrics','feature_select','whitetest']):
        df.to_csv(os.path.join(results_path, f'{filename}_full_{timestamp}.csv'),index_label='Model', float_format='%.3f')

    # Store single metrics separately
    for name in metrics_names['Metrics']:
        columns = [column for column in df_metrics.columns if name in column]
        df_metrics[columns].to_csv(os.path.join(results_path, f'Metrics_{timestamp}_{name}.csv'),index_label='Model', float_format='%.3f')


def get_df_subset(df, label):
    cols = [col for col in df.columns if label in col]
    df_subset = df[cols]
    df_subset.columns = ["_".join(col.split("_")[:-1]) for col in cols]
    return df_subset


def create_file_name_timestamp():
    return "Experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def metrics_to_json(dict_metrics, filename):
    with open(filename, 'w') as f:
        json.dump(dict_metrics, f)