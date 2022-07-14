from scipy import signal as sig
from .filter import Filter


class FilterDirectForm(Filter):
    """
    Signal filter - based on sklearn TransformerMixin. Can be stored to pickle file.
    Options:
        - keep_nans: Filtered signal still keeps NaN values from original signals
        - remove_offset: Remove offset from signal before filtering, apply offset afterwards
    """
    coef_ = [[0], [0]]

    def __init__(self, remove_offset=False, keep_nans=False, **kwargs):
        super().__init__(self, remove_offset=remove_offset, keep_nans=keep_nans)

    def _fit(self, X, y=None, **fit_params):
        self.coef_ = self.calc_coef(X, y, **fit_params)

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
        return [[0], [0]]


