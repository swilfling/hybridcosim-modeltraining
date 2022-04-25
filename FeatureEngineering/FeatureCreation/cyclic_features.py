import numpy as np
import pandas as pd


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


def create_cyclical_features(df, label, T=7):
    df[f'{label}_sin'] = np.sin(df[label] * 2 * np.pi / T)
    df[f'{label}_cos'] = np.cos(df[label] * 2 * np.pi / T)
    return df


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