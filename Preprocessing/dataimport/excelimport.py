import pandas as pd

from . import DataImport


class ExcelImport(DataImport):
    """
    Data import for excel files.
    Supports xls and xlsx format.
    """
    fmt: str = 'xlsx'

    def __init__(self, fmt='xlsx', **kwargs):
        super().__init__(**kwargs)
        self.fmt = fmt

    def read_file(self, filename=""):
        """
        Read excel file
        @return: dataframe
        """
        return pd.read_excel(f'{filename}.{self.fmt}')
