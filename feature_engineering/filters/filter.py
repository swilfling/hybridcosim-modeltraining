from abc import abstractmethod
from scipy import signal as sig

from .nan_comp import NaNComp
from .offset_comp import OffsetComp
from ..transformers.transformer_selectedfeats import Transformer_SelectedFeats


class Filter(Transformer_SelectedFeats):
    """
    Signal filter - based on sklearn TransformerMixin. Can be stored to pickle file (StoreInterface).
    Options:
        - keep_nans: Filtered signal still keeps NaN values from original signals
        - remove_offset: Remove offset from signal before filtering, apply offset afterwards
    """
    keep_nans = False
    remove_offset = False
    offset_comp = None
    nan_comp = None
    coef_ = [[0], [0]]

    def __init__(self, remove_offset=False, keep_nans=False, **kwargs):
        super().__init__(**kwargs)
        self._set_attrs(remove_offset=remove_offset, keep_nans=keep_nans)
        self.offset_comp = OffsetComp()
        self.nan_comp = NaNComp()

    def _fit(self, X, y=None, **fit_params):
        self.coef_ = self._fit_model(X, y, **fit_params)
        return self

    def _transform(self, X):
        """
        Filter signal
        @param x: Input feature vector (n_samples, n_features)
        """
        # Remove offset
        X_to_filter = self.offset_comp.fit_transform(X) if self.remove_offset else X
        # Keep NaN values
        if self.keep_nans:
            X_to_filter = self.nan_comp.fit_transform(X_to_filter)
        # Transform features
        x_filt = self._transform_feats(X_to_filter)
        # Apply NaNs
        if self.keep_nans:
            x_filt = self.nan_comp.inverse_transform(x_filt)
        # Add offset
        if self.remove_offset:
            x_filt = self.offset_comp.inverse_transform(x_filt)
        return x_filt

    def _transform_feats(self, X):
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
    def _fit_model(self, X, y=None, **fit_params):
        """
        Override this method to create filter coeffs.
        """
        raise NotImplementedError
