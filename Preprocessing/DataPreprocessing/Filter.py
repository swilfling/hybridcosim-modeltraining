from sklearn.base import TransformerMixin
import numpy as np
import scipy.signal as sig
import pandas as pd
from abc import abstractmethod
from ModelTraining.datamodels.datamodels.processing.feature_extension.StoreInterface import StoreInterface


class Filter(TransformerMixin, StoreInterface):
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


class ButterworthFilter(Filter):
    """
    Butterworth lowpass filter for data smoothing.
    """
    T = 10
    order = 2

    def __init__(self, T=10, order=2, **kwargs):
        super().__init__(**kwargs)
        self._set_attrs(T=T, order=order)

    def _fit(self, X, y=None, **fit_params):
        return sig.butter(self.order, 1 / self.T)


class ChebyshevFilter(Filter):
    """
       Chebyshev lowpass filter for data smoothing.
    """
    T = 10
    order = 2
    ripple = 0.1

    def __init__(self, T=10, order=2, ripple=0.1, **kwargs):
        super().__init__(**kwargs)
        self._set_attrs(T=T, order=order, ripple=ripple)

    def _fit(self, X, y=None, **fit_params):
        return sig.cheby1(self.order, self.ripple, 1 / self.T)


class Envelope_MA(Filter):
    """
        Envelope detector through moving average (parameter T of moving average filter can be set)
    """
    T = 10
    envelope_h = None
    envelope_l = None
    envelope_avg = None

    def __init__(self, T=10, **kwargs):
        super().__init__(**kwargs)
        self._set_attrs(T=T)

    def _fit(self, X, y=None, **fit_params):
        return None

    def _transform(self, X):
        """
        Calculate envelope.
        This is taken from https://stackoverflow.com/a/69357933
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        X_df = pd.DataFrame(X)
        self.envelope_h = X_df.rolling(window=self.T).max().shift(int(-self.T / 2))
        self.envelope_l = X_df.rolling(window=self.T).min().shift(int(-self.T / 2))
        self.envelope_avg = np.mean(np.dstack((self.envelope_l, self.envelope_h)),axis=-1)
        return self.envelope_avg

    def get_max_env(self):
        """
            Get upper bound of envelope
        """
        return self.envelope_h

    def get_min_env(self):
        """
        Get lower bound of envelope
        """
        return self.envelope_l

