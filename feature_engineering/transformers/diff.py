import pandas as pd
from .basictransformer import BasicTransformer


class Diff(BasicTransformer):
    """
    Differencing transformation
    """
    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X.diff()
        else:
            return pd.DataFrame(X).diff().to_numpy()

    def _get_feature_names_out(self, feature_names=None):
        return [f"{name}_diff" for name in feature_names] if feature_names is not None else None