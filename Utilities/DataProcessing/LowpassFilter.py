from sklearn.base import TransformerMixin
import numpy as np
import scipy.signal as sig


class ButterworthFilter(TransformerMixin):
    """
    Butterworth lowpass filter

    """
    T = 1
    order = 2
    keep_nans = False
    mask_nan = None

    def __init__(self, T=1, order=2, keep_nans=False):
        self._set_attrs(T=T, order=order, keep_nans=keep_nans)
        self.coeffs = sig.butter(self.order, 2 * np.pi / T)

    def _set_attrs(self, **kwargs):
        for name, val in kwargs.items():
            setattr(self, name, val)

    def fit_transform(self, X, y=None, **fit_params):
        if self.keep_nans:
            self.mask_nan = X.isna()
            X = np.nan_to_num(X)
        x_transformed = sig.lfilter(self.coeffs[0], self.coeffs[1], X, axis=0)
        if self.keep_nans:
            x_transformed[self.mask_nan is True] = np.nan
        return x_transformed


