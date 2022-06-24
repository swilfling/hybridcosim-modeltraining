from .. import FeatureSelector
import numpy as np


class FeatureSelectThreshold(FeatureSelector):
    """
        FeatureSelector - based on threshold
        Supports:
            - Thresholding - coef > threshold
            - Omit zero samples from calculation if necessary
    """
    coef_ = None
    nonzero = None
    omit_zero_samples = False
    nz_idx = None
    thresh = 0

    def __init__(self, thresh=0, omit_zero_samples=False,  **kwargs):
        super().__init__(**kwargs)
        self.thresh = thresh
        self.omit_zero_samples = omit_zero_samples

    def _fit(self, X, y=None, **fit_params):
        """
        Fit transformer
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        self.nonzero = ~np.all(X == 0, axis=0)
        self.nz_idx = np.where(self.nonzero)
        self.n_features_in_ = X.shape[-1]
        if self.omit_zero_samples:
            coef = self.calc_coef(X[:, self.nonzero], y, **fit_params)
            self.coef_ = np.zeros(X.shape[-1])
            self.coef_[self.nz_idx] = coef
        else:
            self.coef_ = self.calc_coef(X, y, **fit_params)

    def calc_coef(self, X, y=None, **fit_params):
        """
        Calculate coefficients for feature selection trheshold.
        @param X: input features (n_samples x n_features)
        @param y: target features
        @return: coefficients
        """
        return np.zeros(X.shape[-1])

    def get_coef(self):
        """
        get coefficients for selected features
        @return: coeffs
        """
        if self.coef_ is not None:
            return self.coef_[self.get_support()]

    def _get_support_mask(self):
        """
        Get boolean mask of selected features - override if necessary.
        @return: boolean array
        """
        if self.coef_ is not None:
            return self.coef_ > self.thresh if not self.omit_zero_samples else (self.coef_ > self.thresh) & self.nonzero
