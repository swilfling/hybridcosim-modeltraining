import numpy as np
from sklearn.feature_selection import f_regression

from .feature_selector import FeatureSelector


class FThreshold(FeatureSelector):
    """
    F-Threshold:
    Threshold based on F-test of the Pearson correlation value.
    The F-test values are normalized between 0 and 1 for the smallest to highest value.
    """
    def _fit(self, X, y=None, **fit_params):
        f_val = f_regression(X, y)[0]
        # Normalize f val
        return f_val / np.nanmax(f_val)