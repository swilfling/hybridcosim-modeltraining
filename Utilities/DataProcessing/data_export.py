import json
import os
from datetime import datetime

import pandas as pd



def store_to_csv(df, csv_file_path):
    date_format = '%d.%m.%Y %H:%M'
    if df is not None:
        df.to_csv(csv_file_path, sep=";", index_label="Zeitraum", date_format=date_format)


def store_init_values(df: pd.DataFrame, inputs, csv_output_file):
    init_values = df[inputs].iloc[0]
    init_values.to_csv(csv_output_file, sep=";", line_terminator=os.linesep, index_label="Feature")


def metrics_to_json(dict_metrics, filename):
    with open(filename, 'w') as f:
        json.dump(dict_metrics, f)


def create_df_empty(keys = []):
    index = [f"{key.split('_')[0]} - {'expanded' if key.split('_')[1] == 'PolynomialExpansion' else 'basic'}" for key in
             keys]
    return pd.DataFrame(index=index)


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