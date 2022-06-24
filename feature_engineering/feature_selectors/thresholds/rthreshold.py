import numpy as np
from sklearn.feature_selection import r_regression

from . import FeatureSelectThreshold


class RThreshold(FeatureSelectThreshold):
    """
    R-Threshold:
    Threshold based on absolute value of the Pearson correlation value.
    """
    def calc_coef(self, X, y=None, **fit_params):
        """
        Calculate coefficients for feature selection trheshold.
        @param X: input features (n_samples x n_features)
        @param y: target features
        @return: coefficients
        """
        if X.shape[-1] > 0:
            return np.abs(r_regression(X, y))
        return np.zeros(1)