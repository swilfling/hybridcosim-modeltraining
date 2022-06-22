import numpy as np
from minepy.mine import MINE

from .feature_selector import FeatureSelector


class MICThreshold(FeatureSelector):
    """
    MIC-threshold
    Features are selected based on MIC.
    """
    def _fit(self, X, y=None, **fit_params):
        n_features = X.shape[-1]
        coef = np.zeros(n_features)
        mine = MINE()
        for i in range(n_features):
            mine.compute_score(X[:, i], np.ravel(y))
            coef[i] = mine.mic()
        return coef