import numpy as np
from minepy.mine import MINE

from . import FeatureSelectThreshold


class MICThreshold(FeatureSelectThreshold):
    """
    MIC-threshold
    Features are selected based on MIC.
    """
    def calc_coef(self, X, y=None, **fit_params):
        """
        Calculate coefficients for feature selection trheshold.
        @param X: input features (n_samples x n_features)
        @param y: target features
        @return: coefficients
        """
        n_features = X.shape[-1]
        coef = np.zeros(n_features)
        mine = MINE()
        for i in range(n_features):
            mine.compute_score(X[:, i], np.ravel(y))
            coef[i] = mine.mic()
        return coef