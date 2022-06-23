import numpy as np
from sklearn.base import TransformerMixin

from ModelTraining.feature_engineering.interfaces import PickleInterface


class OffsetComp(TransformerMixin, PickleInterface):
    offset = None

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **fit_params):
        self.offset = np.nanmean(X, axis=0)
        return self

    def transform(self, X):
        return X - self.offset

    def inverse_transform(self, X):
        return X + self.offset