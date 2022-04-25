from sklearn.base import TransformerMixin
import numpy as np
import scipy.signal as sig

from ModelTraining.datamodels.datamodels.processing.feature_extension.StoreInterface import StoreInterface


class ButterworthFilter(TransformerMixin, StoreInterface):
    """
    Butterworth lowpass filter for data smoothing
    Implements sklearn's TransformerMixin interface
    """
    T = 10
    order = 2
    keep_nans = False
    mask_nan = None
    coeffs = None

    def __init__(self, T=10, order=2, keep_nans=False):
        self._set_attrs(T=T, order=order, keep_nans=keep_nans)
        self.coeffs = sig.butter(self.order, 1 / T)

    def fit_transform(self, X, y=None, **fit_params):
        if self.keep_nans:
            self.mask_nan = X.isna()
            X = np.nan_to_num(X)
        x_transformed = sig.lfilter(self.coeffs[0], self.coeffs[1], X, axis=0)
        if self.keep_nans:
            x_transformed[self.mask_nan is True] = np.nan
        return x_transformed


