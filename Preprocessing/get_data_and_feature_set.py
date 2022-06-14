import os
import pandas as pd
import ModelTraining.Preprocessing.DataImport.data_import as data_import
import ModelTraining.Preprocessing.DataPreprocessing.data_preprocessing as dp_utils
from . FeatureCreation.feature_creation import create_additional_feats


def get_data(data_filename):
    extension = data_filename.split('.')[-1]
    filename = list(os.path.split(data_filename))[-1].split('.')[0]
    # Check extension and parse hd5
    if extension == "hd5":
        data = data_import.parse_hdf_solarhouse2(data_filename)
        T_FBH_modulation = 20
        data = dp_utils.demod_signals(data, ['T_FBH_VL', 'T_FBH_RL', 'Vd_FBH'], keep_nans=True, T_mod=T_FBH_modulation)
    elif extension == 'xlsx':
        data = data_import.parse_excel_cps_data(data_filename) if filename == 'cps_data' else data_import.parse_excel_sensor_A6(data_filename)
    else:
        if filename == 'Beyond_B20_full' or filename == 'Beyond_B12_full':
            data = data_import.import_data(data_filename, sep=',',freq='1H', index_col='dt')
            data = data.drop(pd.date_range(pd.Timestamp(2014, 4, 13), pd.Timestamp(2014, 4, 24), freq='1H'), axis=0)
        elif filename == 'T24_full':
            data = data_import.import_data(data_filename, sep=',',freq='1H', index_col='t')
        else:
            # Solarhouse 1
            data = data_import.import_data(data_filename, freq='15T')
            data = data[:pd.Timestamp(2019, 10, 25)]

    data = create_additional_feats(data=data, filename=data_filename)
    return data
