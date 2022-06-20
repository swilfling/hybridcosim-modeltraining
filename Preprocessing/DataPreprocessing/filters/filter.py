from abc import abstractmethod

import numpy as np
from scipy import signal as sig
from sklearn.base import TransformerMixin, _OneToOneFeatureMixin

from ....datamodels.datamodels.wrappers.feature_extension.store_interface import StoreInterface


class Filter(TransformerMixin, StoreInterface, _OneToOneFeatureMixin):
    """
    Signal filter - based on sklearn TransformerMixin. Can be stored to pickle file (StoreInterface).
    Options:
        - keep_nans: Filtered signal still keeps NaN values from original signals
        - remove_offset: Remove offset from signal before filtering, apply offset afterwards
    """
    keep_nans = False
    mask_nan = None
    remove_offset = False
    offset = None
    coef_ = [[0], [0]]

    def __init__(self, remove_offset=False, keep_nans=False, **kwargs):
        self._set_attrs(remove_offset=remove_offset, keep_nans=keep_nans)

    def fit(self, X, y=None, **fit_params):
        self.coef_ = self._fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """
        Filter signal
        @param x: Input feature vector (n_samples, n_features)
        """
        if self.remove_offset:
            self.offset = np.nanmean(X, axis=0)
            X = X - self.offset
        if self.keep_nans:
            self.mask_nan = X.isna()
            X = np.nan_to_num(X)
        x_filt = self._transform(X)
        if self.remove_offset:
            x_filt = x_filt + self.offset
        if self.keep_nans:
            x_filt[self.mask_nan is True] = np.nan
        return x_filt

    def _transform(self, X):
        """
        Filter signal. Override if necessary.
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        return sig.lfilter(*self.coef_, X, axis=0)

    def get_coef(self):
        """
        Get filter coefficients.
        """
        return self.coef_

    @abstractmethod
    def _fit(self, X, y=None, **fit_params):
        """
        Override this method to create filter coeffs.
        """
        raise NotImplementedError