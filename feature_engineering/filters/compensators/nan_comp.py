import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from ...featureengineeringbasic.interfaces import PickleInterface


class NaNComp(TransformerMixin, PickleInterface, BaseEstimator):
    mask_ = None
    keep_nans = True

    def __init__(self, keep_nans=True, **kwargs):
        self.keep_nans = keep_nans

    def fit(self, X, y=None, **fit_params):
        self.mask_ = self.calc_mask(X)
        return self

    def calc_mask(self, X):
        return np.isnan(X)

    def transform(self, X):
        if self.keep_nans:
            return np.nan_to_num(X)
        return X

    def inverse_transform(self, X):
        X_tr = X
        if self.keep_nans:
            X_tr[self.mask_ is True] = np.nan
        return X_tr


