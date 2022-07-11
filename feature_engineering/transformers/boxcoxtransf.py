import numpy as np
import pandas as pd
from scipy.stats import boxcox
from .basictransformer import BasicTransformer


class Boxcox(BasicTransformer):
    """
    Box-cox transformation - supports only samples > 0
    """
    omit_zero_samples = False
    offset = 0.000001

    def __init__(self, omit_zero_samples=False, offset=0.000001, **kwargs):
        self.omit_zero_samples = omit_zero_samples
        self.offset = offset
        super().__init__(**kwargs)

    def transform(self, X):
        if np.any(X < 0):
            raise ValueError("Box-cox transformation only supports positive samples.")

        if isinstance(X, pd.DataFrame):
            X_tr = X.copy()
            for col in X.columns:
                if self.omit_zero_samples:
                    cur_feat = (X_tr[col][X_tr[col] != 0]).to_numpy().flatten()
                    X_tr[col][X_tr[col] != 0] = boxcox(cur_feat)[0]
                else:
                    X_tr[col] = boxcox(X[col] + self.offset)[0]
            return X_tr
        else:
            X_tr = np.expand_dims(X, axis=-1) if X.ndim == 1 else X
            for i in range(X.shape[1]):
                cur_feat = X[:, i]
                if self.omit_zero_samples:
                    cur_feat[cur_feat != 0] = boxcox(cur_feat[cur_feat != 0])[0]
                else:
                    cur_feat = boxcox(cur_feat + self.offset)[0]
                X_tr[:,i] = cur_feat
            return X_tr

    def _get_feature_names_out(self, feature_names=None):
        return [f"{name}_boxcox" for name in feature_names] if feature_names is not None else None