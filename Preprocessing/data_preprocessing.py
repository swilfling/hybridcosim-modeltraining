import numpy as np
import pandas as pd
from ModelTraining.feature_engineering.featureengineering.filters import ChebyshevFilter, Envelope_MA

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


def preprocess_data(data: pd.DataFrame, filename=""):
    """
        This is the main preprocessing function.
        @param data: Data
        @param filename: filename of data
        @return: pre-processed data
    """
    # Solarhouse 1
    if filename == 'Resampled15min':
        data = data[:pd.Timestamp(2019, 10, 25)]
        data = inverse_transf(data, 'VDSolar')
        for feature in ['TAussen', 'SGlobal', 'TSolarRL']:
            data = feature_mult(data, feature, 'VDSolar_inv')

    # Solarhouse 2
    if filename == 'P2017_20_Solarhouse2':
        T_FBH_modulation = 20
        data["DeltaTSolar"] = data["T_Solar_VL"] - data["T_Solar_RL"]
        labels = ['T_FBH_VL', 'T_FBH_RL', 'Vd_FBH']
        filt_labels = [f'{label}_filt' for label in labels]
        env_labels = [f'{label}_env' for label in labels]
        env_det = Envelope_MA(T=T_FBH_modulation, keep_nans=True, remove_offset=True)
        cheb_filt = ChebyshevFilter(T=T_FBH_modulation * 10, keep_nans=True, remove_offset=True)
        data[filt_labels] = cheb_filt.fit_transform(data[labels])
        data[env_labels] = env_det.fit_transform(data[labels])

    # Inffeldgasse
    if filename in ['cps_data', 'sensor_A6', 'sensor_B2', 'sensor_C6']:
        data = holiday_weekend(data)

    # Beyond
    if filename in ['Beyond_B20_full', 'Beyond_B12_full']:
        data = data.drop(pd.date_range(pd.Timestamp(2014, 4, 13), pd.Timestamp(2014, 4, 24), freq='1H'), axis=0)
        if filename == 'Beyond_B12_full':
            data = feature_mean(data, ['TB12BR1', 'TB12BR2', 'TB12BR3', 'TB12LR'], 'TB12')
        if filename == 'Beyond_B20_full':
            data = feature_mean(data, ['TB20BR1', 'TB20BR2', 'TB20BR3', 'TB20LR'], 'TB20')
            data['SGlobalH_wo_lum_LR'] = data['SGlobalH'] - data['lumB20LR'] # TODO add this also for other datsetrs

        data['SGlobalH_TAmbient'] = data['SGlobalH'] * data['TAmbient']
        data['SGlobalH_TAmbient_vWind_inv'] = data['SGlobalH_TAmbient'] / data['vWind']
        data['SGlobalH_TAmbient_vWind_inv'][data['vWind'] == 0] = 0


    data = data.dropna(axis=0)
    data = data.astype('float')
    return data

