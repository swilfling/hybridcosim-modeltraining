from . import Transformer_inplace
import pandas as pd


class Diff(Transformer_inplace):
    """
    Differencing transformation
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.diff()
        else:
            return pd.DataFrame(X).diff().to_numpy()

