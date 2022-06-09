import numpy as np
from sklearn.feature_selection import r_regression

from .feature_selector import FeatureSelector


class RThreshold(FeatureSelector):
    """
    R-Threshold:
    Threshold based on absolute value of the Pearson correlation value.
    """
    def _fit(self, X, y=None, **fit_params):
        if X.shape[-1] > 0:
            return np.abs(r_regression(X, y))
        return None