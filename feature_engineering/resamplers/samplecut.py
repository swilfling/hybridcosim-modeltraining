from imblearn.base import SamplerMixin, BaseSampler
from sklearn.base import BaseEstimator
from ..featureengineeringbasic.interfaces import BaseFitTransform, FeatureNames


class SampleCut_imblearn(BaseSampler, FeatureNames):
    num_samples = 0
    _sampling_type = 'bypass'

    def __init__(self, num_samples=0, **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples

    def fit_resample(self, X, y=None):
        if y is not None:
            return X[self.num_samples:], y[self.num_samples:]
        else:
            return X[self.num_samples:], None

    def _fit_resample(self, X, y):
        if y is not None:
            return X[self.num_samples:], y[self.num_samples:]
        else:
            return X[self.num_samples:], None


class SampleCut(BaseEstimator, BaseFitTransform, FeatureNames):
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