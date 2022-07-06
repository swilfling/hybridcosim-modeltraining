from scipy import signal as sig
import pandas as pd
import numpy as np

from sklearn.base import TransformerMixin, BaseEstimator
from .compensators import OffsetComp, NaNComp
from ..interfaces import BaseFitTransform, MaskFeats, PickleInterface


class Filter(MaskFeats, BaseFitTransform, PickleInterface, TransformerMixin, BaseEstimator):
    """
    Signal filter - based on sklearn TransformerMixin. Can be stored to pickle file.
    Options:
        - keep_nans: Filtered signal still keeps NaN values from original signals
        - remove_offset: Remove offset from signal before filtering, apply offset afterwards
    """
    offset_comp_ = None
    nan_comp_ = None
    coef_ = [[0], [0]]
    remove_offset = False
    keep_nans = False

    def __init__(self, remove_offset=False, keep_nans=False, features_to_transform=None, **kwargs):
        MaskFeats.__init__(self, features_to_transform=features_to_transform)
        self.remove_offset = remove_offset
        self.keep_nans = keep_nans

    def fit(self, X, y=None, **fit_params):
        self.offset_comp_ = OffsetComp(self.remove_offset)
        self.nan_comp_ = NaNComp(self.keep_nans)
        X_to_filter = self.nan_comp_.fit_transform(self.offset_comp_.fit_transform(self.mask_feats(X)))
        return super().fit(X_to_filter)

    def _fit(self, X, y=None, **fit_params):
        self.coef_ = self.calc_coef(X, y, **fit_params)

    def transform(self, X):
        """
        Filter signal
        @param x: Input feature vector (n_samples, n_features)
        """
        X_masked = self.mask_feats(X)
        # Remove offset and NaNs
        X_to_filter = self.nan_comp_.fit_transform(self.offset_comp_.fit_transform(X_masked))
        # Transform features
        x_filt = self._transform(X_to_filter)
        if isinstance(X_to_filter, pd.DataFrame):
            x_filt = pd.DataFrame(index=X_to_filter.index, columns=X_to_filter.columns, data=x_filt)
        if isinstance(X_to_filter, pd.Series):
            x_filt = pd.Series(index=X_to_filter.index, data=np.ravel(x_filt))
        # Apply NaNs and offset
        x_filt = self.offset_comp_.inverse_transform(self.nan_comp_.inverse_transform(x_filt))
        return self.combine_feats(x_filt, X)

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

    def calc_coef(self, X, y=None, **fit_params):
        """
        Override this method to create filter coeffs.
        """
        return None
