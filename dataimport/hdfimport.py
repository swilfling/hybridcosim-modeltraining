import numpy as np
import pandas as pd

from . import DataImport


class HDFImport(DataImport):
    """
    Data import for HDF5 files.
    """
    def read_file(self, filename=""):
        """
         Get dataframe from file
         @return: df
         """
        df = pd.read_hdf(f'{filename}.hd5')
        return df.astype('float')

    def fill_missing_vals(self, df: pd.DataFrame):
        """
            Fill missing values
            @param df: dataframe to modify
            @return: modified df
        """
        df = df.reindex(pd.DatetimeIndex(pd.date_range(df.index[0], df.index[-1], freq='min')), fill_value=np.nan)
        df = df.resample('15min').first()
        df = df.where(df.isna() == False)
        df = df.copy().groupby(df.index.time).ffill() # TODO check this
        return df
