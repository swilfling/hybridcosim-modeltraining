import numpy as np
from sklearn.feature_selection import f_regression

from . import FeatureSelectThreshold


class FThreshold(FeatureSelectThreshold):
    """
    F-Threshold:
    Threshold based on F-test of the Pearson correlation value.
    The F-test values are normalized between 0 and 1 for the smallest to highest value.
    """
    def calc_coef(self, X, y=None, **fit_params):
        """
        Calculate coefficients for feature selection trheshold.
        @param X: input features (n_samples x n_features)
        @param y: target features
        @return: coefficients
        """
        f_val = f_regression(X, y)[0]
        # Normalize f val
        return f_val / np.nanmax(f_val)