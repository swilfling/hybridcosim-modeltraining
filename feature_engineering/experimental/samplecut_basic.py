from sklearn.base import BaseEstimator
from ..featureengineering.interfaces import BaseFitTransform, FeatureNames


class SampleCut_Basic(BaseEstimator, BaseFitTransform, FeatureNames):
    num_samples = 0

    def __init__(self, num_samples=0, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples

    def _fit(self, X, y=None, **fit_params):
        pass

    def _transform(self, X):
        return X[self.num_samples:]

    def inverse_transform(self, X):
        return X