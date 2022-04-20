import os
import json
from datetime import datetime

from math import ceil

import numpy as np
import pandas as pd
import scipy

import ModelTraining.Utilities.dataframe_utils as df_utils
from ModelTraining.Utilities.DataProcessing import signal_processing_utils as sigutils
from ModelTraining.Utilities.DataProcessing.feature_creation import add_cyclical_features, holiday_weekend
from ModelTraining.Utilities.feature_set import FeatureSet


def import_data(csv_input_file="Resampled15min.csv", freq='15T', sep=';',info=False, index_col='Zeitraum'):
    data = pd.read_csv(csv_input_file, sep=sep, encoding='latin-1', header=0, low_memory=False)
    data = df_utils.df_set_date_index(data, index_col)
    if info:
        data.info()
        data.dtypes
    # data preprocessing if there are nulls
    data = data.reindex(pd.date_range(data.index[0], data.index[-1], freq=freq), fill_value=np.nan)
    ### Inverse value for VD_Solar
    if data.get('VDSolar') is not None:
        data['VDSolar_inv'] = 1.0 / data['VDSolar']
        data['VDSolar_inv'][data['VDSolar'] == 0] = 0

    # Resample every 15 min
    return data.resample(freq).first()


def parse_excel_files(WorkingDirectory, extension):
    filenames = os.listdir(WorkingDirectory)
    print(filenames)

    all_data = pd.DataFrame()
    # Assigns which label offers data in what resolution
    # gather those sensors in the same folder
    for filename in filenames:
        if filename.endswith(extension):
            print(filename)
            full_xls_path = os.path.join(WorkingDirectory, filename)
            xls = pd.ExcelFile(full_xls_path)
            for sheet in xls.sheet_names:
                print(f'{sheet}')
                df = pd.read_excel(xls, header=3, sheet_name=sheet)
                df = df.set_index(pd.to_datetime(df['Datum'] + ' ' + df['Zeit'], format='%d.%m.%Y %H:%M:%S'))
                # At this stage you can save every sheet individual csv.
                # df.to_csv(sheet+'.csv', index=True)
                # Sensor units
                unit_column_name = "Einheit"
                unitSensor = df[unit_column_name][df[unit_column_name].notna()][0]  # take the unit
                df = df.rename(columns={'Wert': sheet + '_' + unitSensor},
                               inplace=False)  # assign columns Wert the codename and the unit
                df = df.drop(['Datum', 'Zeit', unit_column_name], axis=1)
                all_data[sheet + '_' + unitSensor] = df[sheet + '_' + unitSensor]

    return all_data

def parse_excel(file, index_col="datetime"):
    df = pd.read_excel(file)
    df = df.rename({'time':'daytime'},axis=1)
    index = pd.Index(pd.to_datetime(df[index_col]))
    df = df.set_index(index)
    df = df.drop([index_col], axis=1)
    return df


def parse_excel_cps_data(file):
    df = parse_excel(file, index_col="datetime")
    df = df.drop(df.columns[0], axis=1)
    df = holiday_weekend(df)
    return df


def parse_excel_sensor_A6(file):
    df = parse_excel(file, index_col="datetime")
    df = df.rename({df.columns[0]:'energy'},axis=1)
    df = holiday_weekend(df)
    return df


def parse_hdf_solarhouse2(filename, keep_nans=False):
    df = parse_hdf(filename)
    df = df_utils.remove_spaces_from_labels(df)

    ############ Rename labels #########################################
    df = df.rename(columns={'T_P_oo': 'T_P_top'})

    ############ Process modulated signals ######################################
    fbh_labels = ['T_FBH_VL', 'T_FBH_RL', 'Vd_FBH']
    for label in fbh_labels:
        signal = df[label]
        T_FBH_modulation = 20
        sig_filtered = sigutils.filter_signal(signal, T_FBH_modulation, keep_nans=keep_nans)
        sig_env = sigutils.calc_avg_envelope(signal, T_FBH_modulation, keep_nans=keep_nans)[0]
        df[f"{label}_filt"] = sig_filtered
        df[f"{label}_env"] = sig_env

    ######## Add columns
    df["DeltaTSolar"] = df["T_Solar_VL"] - df["T_Solar_RL"]

    # Drop columns
    df = df.drop(["P_Recool"], axis=1)

    ############## Fill empty time values ###############################################
    generatetime = pd.DataFrame(columns=['NULL'], index=pd.date_range(df.index[0], df.index[-1], freq='min'))
    df = df.reindex(generatetime.index, fill_value=np.nan)
    df = df.resample('15min').first()
    '''Filling the missing data'''
    if keep_nans == False:
        df = df.where(df.isna() == False)
        df = df.copy().groupby(df.index.time).ffill()

    csv_filename = str.replace(filename, ".hd5", ".csv")

    ### Inverse value for VD_Solar
    df['Vd_Solar_inv'] = 1.0 / df['Vd_Solar']
    df['Vd_Solar_inv'][df['Vd_Solar'] == 0] = 0

    # Solar collector calc
    df['T_Aussen_Vd_Solar_inv'] = df['Vd_Solar_inv'] * df['T_Aussen']
    df['R_Global_Vd_Solar_inv'] = df['Vd_Solar_inv'] * df['R_Global']
    df['T_Solar_RL_Vd_Solar_inv'] = df['Vd_Solar_inv'] * df['T_Solar_RL']

    df.to_csv(csv_filename, sep=';', index=True, header=True, index_label='Zeitraum')
    return df


def parse_hdf(filename):
    return pd.read_hdf(filename)


def load_from_json(filename):
    with open(filename) as f:
        return json.load(f)

def create_statistical_features(df, features_to_select=[],
                                statistical_features=['tmean', 'tstd', 'tmax', 'tmin', 'skew', 'moment', 'kurtosis'],
                                window_size=2, window_type="static"):
    df_slices = ceil(len(df) / window_size)

    for df_slice in range(0, df_slices):
        start = df_slice * window_size
        end = start + window_size
        for col in features_to_select:
            for idx, feat in enumerate(statistical_features):
                if window_type=="static":
                    if f'{col}_{feat}' in df.columns:
                        df[f'{col}_{feat}'][start:end] = getattr(scipy.stats, statistical_features[idx])(df[col][start:end].to_numpy())
                    else:
                        df[f'{col}_{feat}'] = getattr(scipy.stats, statistical_features[idx])(df[col][start:end].to_numpy())
                        print("df[f'{col}_{feat}'] ", df[f'{col}_{feat}'] )
                else:
                    # if feat == "tmax":
                    zscore = lambda x: getattr(scipy.stats, statistical_features[idx])(x)
                    # zscore = lambda x: scipy.stats.tmean(x)
                    # df[f'{col}_{feat}'] = df[col].rolling(2).apply(getattr(scipy.stats, statistical_features[idx]))
                    df[f'{col}_{feat}'] = df[col].rolling(window_size).apply(zscore).fillna(0)
                    
                    
                    # df[f'{col}_{feat}'] = df[col].rolling(window_size).mean()
            df[f'{col}_ptop'] = df[f'{col}_tmax'] - df[f'{col}_tmin']
            mean = df[f'{col}_tmean'].mask(df[f'{col}_tmean']==0).fillna(1.0)
            df[f'{col}_if'] = df[f'{col}_tmax'] // mean
    return df

def create_cyclical_features(df, features_to_select=[]):
    pass

def get_data_and_feature_set(data_filename, interface_filename):
    extension = data_filename.split('.')[-1]
    filename = get_filename(data_filename).split('.')[0]
    # Check extension and parse hd5
    if extension == "hd5":
        data = parse_hdf_solarhouse2(data_filename)
    elif extension == 'xlsx':
        data = parse_excel_cps_data(data_filename) if filename == 'cps_data' else parse_excel_sensor_A6(data_filename)
    else:
        if filename == 'Beyond_B20_full' or filename == 'Beyond_B12_full':
            data = import_data(data_filename, sep=',',freq='1H', index_col='dt')
            data = data.drop(df_utils.create_date_range(2014, 4, 13, 2014, 4, 24), axis=0)
            if filename == 'Beyond_B20_full':
                data['TB20'] = np.mean(data[['TB20BR1','TB20BR2','TB20BR3','TB20LR']], axis=1)
            if filename == 'Beyond_B12_full':
                data['TB12'] = np.mean(data[['TB12BR1', 'TB12BR2', 'TB12BR3', 'TB12LR']], axis=1)
        else:
            data = import_data(data_filename)
            data = data[:pd.Timestamp(2019, 10, 25)]

    data = add_cyclical_features(data)
    data = data.dropna(axis=0)
    data = data.astype('float')
    feature_set = FeatureSet(interface_filename)
    return data, feature_set


def create_file_name_timestamp():
    return "Experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")

def get_filename(src_path):
    return list(os.path.split(src_path))[-1]


