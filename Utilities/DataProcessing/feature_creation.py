from math import ceil

import numpy as np
import pandas as pd
import scipy


def add_cyclical_features(data):
    data["day"] = data.index[:].day
    data['weekday'] = data.index.weekday
    data['daytime'] = data.index.hour
    data["month"] = data.index[:].month
    data = onehot_hours(data, label='daytime')
    data = onehot_months(data, label='month')
    data = onehot_days(data,'day')
    data = onehot_weekdays(data, feature_label='weekday')
    data = create_cyclical_features(data,'weekday',7)
    data = create_cyclical_features(data,'daytime',24)
    return data


def onehot_weekdays(df, feature_label='weekday',weekday_labels=['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']):
    for weekday, day in zip(weekday_labels, range(7)):
        df[weekday] = pd.to_numeric(df[feature_label] == day)
    return df


def onehot_hours(df, label='daytime', div_fact=1):
    range_ = div_fact * np.array(range(int(24/div_fact)))
    for time in range_:
        df[f'hour_{time}'] = pd.to_numeric(df[label] < time+div_fact) * pd.to_numeric(df[label] >= time)
    return df


def onehot_months(df, label='month'):
    range_ = np.array(range(12))
    for time in range_:
        df[f'month_{time}'] = pd.to_numeric(df[label] < time+1) * pd.to_numeric(df[label] >= time)
    return df


def onehot_days(df, label='day'):
    range_ = np.array(range(31))
    for time in range_:
        df[f'day_{time}'] = pd.to_numeric(df[label] < time+1) * pd.to_numeric(df[label] >= time)
    return df


def create_statistical_features(df, features_to_select=[],
                                statistical_features=['tmean', 'tstd', 'tmax', 'tmin', 'skew', 'moment', 'kurtosis'],
                                window_size=2):
    df_slices = ceil(len(df) / window_size)
    for df_slice in range(0, df_slices):
        start = df_slice * window_size
        end = start + window_size
        for col in features_to_select:
            for idx, feat in enumerate(statistical_features):
                if f'{col}_{feat}' in df.columns:
                    df[f'{col}_{feat}'][start:end] = getattr(scipy.stats, statistical_features[idx])(df[col][start:end].to_numpy())
                else:
                    df[f'{col}_{feat}'] = getattr(scipy.stats, statistical_features[idx])(df[col][start:end].to_numpy())
    return df


def create_cyclical_features(df, label, T=7):
    df[f'{label}_sin'] = np.sin(df[label] * 2 * np.pi / T)
    df[f'{label}_cos'] = np.cos(df[label] * 2 * np.pi / T)
    return df


def holiday_weekend(data, holiday_label='holiday', weekday_label='weekday'):
    data['holiday_weekend'] = np.logical_or(data[holiday_label], (data[weekday_label]) == 5 + (data[weekday_label] == 6))
    return data