import numpy as np
import pandas as pd
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
        if X.ndim != 2:
            raise ValueError('MIC Threshold currently only supports two dimensional arrays.')
        n_features = X.shape[-1]
        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same number of samples.')
        coef = np.zeros(n_features)
        mine = MINE()
        for i in range(n_features):
            if isinstance(X, pd.DataFrame):
                mine.compute_score(X.to_numpy()[:, i], np.ravel(y))
            else:
                mine.compute_score(X[:, i], np.ravel(y))
            coef[i] = mine.mic()
        return coef