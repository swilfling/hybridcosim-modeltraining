import json
import os
import numpy as np
import pandas as pd
from typing import List
from ..feature_engineering.interfaces import JSONInterface


class DataImport(JSONInterface):
    """
    Data import - abstract base class.
    Main method:
    - import_data
    Load files:
    - read file - override this!
    Dataframe operations:
    - set index
    - fill missing values
    - df_info
    - rename columns
    - drop columns
    - to csv
    """
    freq: str = "1H"
    index_col: str = ""
    index_type: str = "datetime"
    cols_to_rename: dict = {}
    cols_to_drop: List[str] = []

    def __init__(self, freq='15T', index_col="daytime", cols_to_rename={}, cols_to_drop=[], **kwargs):
        super(DataImport, self).__init__(**kwargs)
        self.freq = freq
        self.index_col = index_col
        self.cols_to_rename = cols_to_rename
        self.cols_to_drop = cols_to_drop

    def import_data(self, filename=""):
        """
        Import data.
        @return: pd.Dataframe
        """
        df = self.read_file(filename)
        df = self.set_index(df)
        df = self.fill_missing_vals(df)
        df = self.rename_columns(df)
        df = self.drop_columns(df)
        return df

    def set_index(self, df: pd.DataFrame):
        """
        Set dataframe index to selected column
        Creates datetime index if necessary
        @param df: dataframe to modify
        @return: modified df
        """
        if self.index_col in df.columns:
            if self.index_type == "datetime":
                df[self.index_col] = pd.to_datetime(df[self.index_col], dayfirst=True)
            df = df.set_index(self.index_col, drop=True)
        return df

    def read_file(self, filename=""):
        """
        Get dataframe from file - override this!
        @return: df
        """
        return pd.DataFrame()

    def fill_missing_vals(self, df: pd.DataFrame):
        """
            Fill missing values
            @param df: dataframe to modify
            @return: modified df
        """
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq=self.freq), fill_value=np.nan)
            return df.resample(self.freq).first()
        return df

    def rename_columns(self, df: pd.DataFrame):
        """
            Rename columns
            @param df: dataframe to modify
            @return: modified df
        """
        return df.rename(self.cols_to_rename, axis=1)

    def drop_columns(self, df: pd.DataFrame):
        """
            Drop columns
            @param df: dataframe to modify
            @return: modified df
        """
        return df.drop(self.cols_to_drop, axis=1)

    def to_csv(self, df: pd.DataFrame, filename=""):
        """
            Store to csv
            @param df: dataframe to store
        """
        df.to_csv(f'{filename}.csv', index_label=self.index_col)

    @staticmethod
    def df_info(df: pd.DataFrame):
        """
            Get info about df
            @param df: dataframe to get info of
        """
        if df is not None:
            df.info()
            print(df.dtypes)


    @staticmethod
    def get_filename(filename: str = ""):
        return list(os.path.split(filename))[-1].split(".")[0]


def load_from_json(filename):
    with open(filename) as f:
        return json.load(f)