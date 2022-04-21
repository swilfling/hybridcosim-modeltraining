from math import ceil

import numpy as np
import pandas as pd
import scipy


def add_cyclical_features(data):
    data["day"] = data.index[:].day
    data['weekday'] = data.index.weekday
    data['daytime'] = data.index.hour
    data["month"] = data.index[:].month
    data = onehot(data, 'daytime', 24, div_fact=1, dst_label='hour')
    data = onehot(data, 'month', 12)
    data = onehot(data, 'day', 31)
    data = onehot_weekdays(data, feature_label='weekday')
    data = create_cyclical_features(data, 'weekday', 7)
    data = create_cyclical_features(data, 'daytime', 24)
    return data


def onehot_weekdays(df, feature_label='weekday',weekday_labels=['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']):
    for weekday, day in zip(weekday_labels, range(7)):
        df[weekday] = pd.to_numeric(df[feature_label] == day)
    return df

def onehot(df, label='month', timerange=31, div_fact=1, dst_label=None):
    if dst_label is None:
        dst_label = label
    for time in np.arange(0,timerange, div_fact):
        df[f'{dst_label}_{time}'] = pd.to_numeric(df[label] < time+div_fact) * pd.to_numeric(df[label] >= time)
    return df


def create_statistical_features(df, features_to_select=[],
                                statistical_features=['tmean', 'tstd', 'tmax', 'tmin', 'skew', 'moment', 'kurtosis'],
                                window_size=2, window_type="static"):
    df_slices = ceil(len(df) / window_size)

    for df_slice in range(0, df_slices):
        start = df_slice * window_size
        end = start + window_size
        for col in features_to_select:
            for idx, feat in enumerate(statistical_features):
                if window_type == "static":
                    if f'{col}_{feat}' in df.columns:
                        df[f'{col}_{feat}'][start:end] = getattr(scipy.stats, statistical_features[idx])(
                            df[col][start:end].to_numpy())
                    else:
                        df[f'{col}_{feat}'] = getattr(scipy.stats, statistical_features[idx])(
                            df[col][start:end].to_numpy())
                       # print("df[f'{col}_{feat}'] ", df[f'{col}_{feat}'])
                else:
                    # if feat == "tmax":
                    zscore = lambda x: getattr(scipy.stats, statistical_features[idx])(x)
                    # zscore = lambda x: scipy.stats.tmean(x)
                    # df[f'{col}_{feat}'] = df[col].rolling(2).apply(getattr(scipy.stats, statistical_features[idx]))
                    df[f'{col}_{feat}'] = df[col].rolling(window_size).apply(zscore).fillna(0)

                    # df[f'{col}_{feat}'] = df[col].rolling(window_size).mean()
            df[f'{col}_ptop'] = df[f'{col}_tmax'] - df[f'{col}_tmin']
            mean = df[f'{col}_tmean'].mask(df[f'{col}_tmean'] == 0).fillna(1.0)
            df[f'{col}_if'] = df[f'{col}_tmax'] // mean
    return df


def create_cyclical_features(df, label, T=7):
    df[f'{label}_sin'] = np.sin(df[label] * 2 * np.pi / T)
    df[f'{label}_cos'] = np.cos(df[label] * 2 * np.pi / T)
    return df


def holiday_weekend(data, holiday_label='holiday', weekday_label='weekday'):
    data['holiday_weekend'] = np.logical_or(data[holiday_label], (data[weekday_label]) == 5 + (data[weekday_label] == 6))
    return data


def add_cycl_feats(dict_usecase, feature_set):
    hour_1hot_feat = [f'hour_{h}' for h in range(24)]
    weekday_1hot_feat = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    # Add one hot encoded features and cyclic features
    feature_set.add_static_input_features(dict_usecase.get('cyclical_feats', []),
                                          feature_set.get_output_feature_names())
    #print(dict_usecase.get('onehot_feats'))
    if 'hours' in dict_usecase.get('onehot_feats', []):
        feature_set.add_static_input_features(hour_1hot_feat, feature_set.get_output_feature_names())
    if 'weekdays' in dict_usecase.get('onehot_feats', []):
        feature_set.add_static_input_features(weekday_1hot_feat, feature_set.get_output_feature_names())


def add_stat_feats(data, dict_usecase, feature_set):
    # Add statistical features
    # Adding statistical features, default are: MEAN, STD, MAX, MIN, 'SKEWNESS', 'MOMENT', 'KURTOSIS' over the
    #    defined window size
    stat_feats = dict_usecase.get('stat_feats', [])
    print("Creating statistical features")
    data = create_statistical_features(data, features_to_select=stat_feats,
                                                   window_size=dict_usecase.get('stat_ws', 24))
    statistical_feature_names = [f"{name}_{stat_name}" for name in stat_feats for stat_name in
                                 dict_usecase.get('stat_vals', [])]
    feature_set.add_static_input_features(statistical_feature_names, feature_set.get_output_feature_names())
    return data, feature_set
