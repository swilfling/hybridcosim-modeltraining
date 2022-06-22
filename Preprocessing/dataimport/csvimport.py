import pandas as pd

from . import DataImport


class CSVImport(DataImport):
    """
    CSV data import.
    Supports: different separators
    """
    sep: str = ","

    def __init__(self, sep=',', **kwargs):
        super().__init__(**kwargs)
        self.sep = sep

    def read_file(self, filename=""):
        """
         Get dataframe from file
         @return: df
         """
        return pd.read_csv(f'{filename}.csv', sep=self.sep)
