import os

import pandas as pd
from matplotlib import pyplot as plt
from ModelTraining.Utilities.Parameters import TrainingResults


def env_max(maxvals, windowsize):
    maxvals_downsampled = maxvals.rolling(window=windowsize).max().drop_duplicates()
    maxvals_downsampled[maxvals.index[0]] = maxvals.iloc[0]
    maxvals_downsampled[maxvals.index[-1]] = maxvals.iloc[-1]
    return maxvals_downsampled.reindex(maxvals.index).interpolate()


def env_min(maxvals, windowsize):
    maxvals_downsampled = maxvals.rolling(window=windowsize).min().drop_duplicates()
    maxvals_downsampled[maxvals.index[0]] = maxvals.iloc[0]
    maxvals_downsampled[maxvals.index[-1]] = maxvals.iloc[-1]
    return maxvals_downsampled.reindex(maxvals.index).interpolate()


def get_result_df(path, model_types, baseline_model_type, target_val, expansion):
    dict_expansion_names = {'IdentityExpander': 'basic features', 'PolynomialExpansion': 'Polyfeatures'}
    results_baseline = TrainingResults.load_pkl(path, f'results_{baseline_model_type}_{target_val}_IdentityExpander.pkl')
    df = results_baseline.test_result_df(col_names=['Measurement value', baseline_model_type])
    for model_type in model_types:
        for expansion_set in expansion:
            results_new = TrainingResults.load_pkl(path, f'results_{model_type}_{target_val}_{expansion_set[-1]}.pkl')
            # TODO: support for multi-column
            df[f'{model_type} - {dict_expansion_names[expansion_set[-1]]}'] = results_new.test_pred_vals()
    return df


def get_df(path, model_types, baseline_model_type, target_val, expansion):
    dict_expansion_names = {'IdentityExpander': 'basic features', 'PolynomialExpansion': 'Polyfeatures'}
    df = pd.read_csv(os.path.join(path, f'Timeseries_{baseline_model_type}_{target_val}_IdentityExpander.csv'))
    df.index = pd.DatetimeIndex(df[df.columns[0]])
    df = df.drop(df.columns[0], axis=1)
    df = df.rename({f'predicted_{target_val}':baseline_model_type, target_val:'Measurement value'},axis=1)
    for model_type in model_types:
        for expansion_set in expansion:
            df_new = pd.read_csv(os.path.join(path, f'Timeseries_{model_type}_{target_val}_{expansion_set[-1]}.csv'))
            df_new.index = pd.DatetimeIndex(df_new[df_new.columns[0]])
            df = df.join(df_new[f'predicted_{target_val}'])
            df = df.rename({f'predicted_{target_val}':f'{model_type} - {dict_expansion_names[expansion_set[-1]]}'},axis=1)
    return df


def plot_line(x, ymax,color, label):
    plt.plot([x,x], [0,ymax], color=color, label=label)

