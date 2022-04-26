import os
import json

import numpy as np
import pandas as pd


def import_data(csv_input_file="Resampled15min.csv", freq='15T', sep=';',info=False, index_col='Zeitraum'):
    data = pd.read_csv(csv_input_file, sep=sep, encoding='latin-1', header=0, low_memory=False)
    data[index_col] = pd.to_datetime(data[index_col], dayfirst=True)
    data = data.set_index(index_col)
    if info:
        data.info()
        data.dtypes
    # data preprocessing if there are nulls
    data = data.reindex(pd.date_range(data.index[0], data.index[-1], freq=freq), fill_value=np.nan)
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
    return df


def parse_excel_sensor_A6(file):
    df = parse_excel(file, index_col="datetime")
    df = df.rename({df.columns[0]:'energy'},axis=1)
    return df


def parse_hdf_solarhouse2(filename, keep_nans=False):
    df = pd.read_hdf(filename)
    df = df.rename(columns={label: label.split(" ")[0] for label in df.columns})

    df = df.astype('float')
    ############ Rename labels #########################################
    df = df.rename(columns={'T_P_oo': 'T_P_top'})

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
    df.to_csv(csv_filename, sep=';', index=True, header=True, index_label='Zeitraum')
    return df


def load_from_json(filename):
    with open(filename) as f:
        return json.load(f)


