import ennemi
import numpy as np

from . import FeatureSelectThreshold


class EnnemiThreshold(FeatureSelectThreshold):
    """
    ennemi-threshold
    Features are selected based on ennemi criterion.
    """
    def calc_coef(self, X, y=None, **fit_params):
        """
        Calculate coefficients for feature selection trheshold.
        @param X: input features (n_samples x n_features)
        @param y: target features
        @return: coefficients
        """
        vals = [ennemi.estimate_corr(np.ravel(y), X[:,i], preprocess=True) for i in range(X.shape[-1])]
        return np.array(vals).ravel()