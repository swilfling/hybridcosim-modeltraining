import numpy as np
from ..featureengineering.interfaces import BasicTransformer


class SampleReInit(BasicTransformer):
    num_samples = 0
    init_val = 0
    input_shape_= None

    def __init__(self, num_samples=0, init_val=0, **kwargs):
        self.num_samples = num_samples
        self.init_val = init_val

    def _fit(self, X, y, **fit_params):
        self.input_shape_ = X.shape

    def transform(self, X):
        X[self.num_samples:] = self.init_val
        return X

    def inverse_transform(self, X):
        # return X[self.num_samples:]
        return X

    def create_lookback_samples(self):
        return np.ones((self.num_samples, *self.input_shape_[1:])) * self.init_val if self.input_shape_ is not None else None