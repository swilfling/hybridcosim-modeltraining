import numpy as np
from sklearn.base import TransformerMixin

from ModelTraining.feature_engineering.interfaces import PickleInterface


class NaNComp(TransformerMixin, PickleInterface):
    mask_nan = None

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **fit_params):
        self.mask_nan = np.isnan(X)
        return self

    def transform(self, X):
        return np.nan_to_num(X)

    def inverse_transform(self, X):
        X_tr = X
        X_tr[self.mask_nan is True] = np.nan
        return X_tr