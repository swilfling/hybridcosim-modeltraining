import numpy as np


def inverse_transf(data, label):
    data[f'{label}_inv'] = 1.0 / data[label]
    data[f'{label}_inv'][data[label] == 0] = 0
    return data


def feature_mult(data, label_1, label_2):
    data[f'{label_1}_{label_2}'] = data[label_1] * data[label_2]
    return data


def feature_mean(data, labels, output_label):
    data[output_label] = np.mean(data[labels], axis=1)
    return data


def holiday_weekend(data, holiday_label='holiday', weekday_label='weekday'):
    data['holiday_weekend'] = np.logical_or(data[holiday_label], (data[weekday_label]) == 5 + (data[weekday_label] == 6))
    return data


def create_additional_feats(data, filename):
    if "P2017_20_Solarhouse_2" in filename:
        data = inverse_transf(data, 'Vd_Solar')
        for feature in ['T_Aussen','R_Global','T_Solar_RL']:
            data = feature_mult(data, feature, 'Vd_Solar_inv')
    if 'Beyond_B20_full' in filename:
        data = feature_mean(data, ['TB20BR1','TB20BR2','TB20BR3','TB20LR'], 'TB20')
    if 'Beyond_B12_full' in filename:
        data = feature_mean(data,['TB12BR1', 'TB12BR2', 'TB12BR3', 'TB12LR'],'TB12')
    if "Resampled15min" in filename:
        data = inverse_transf(data, 'VDSolar')
        for feature in ['TAussen','SGlobal','TSolarRL']:
            data = feature_mult(data, feature, 'VDSolar_inv')
    if 'cps_data' in filename or 'sensor_A6' in filename or 'sensor_B2' in filename or 'sensor_C6' in filename:
        data = holiday_weekend(data)
    data = data.dropna(axis=0)
    data = data.astype('float')
    return data


