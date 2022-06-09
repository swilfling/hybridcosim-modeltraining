import ennemi
import numpy as np

from .feature_selector import FeatureSelector


class EnnemiThreshold(FeatureSelector):
    """
    ennemi-threshold
    Features are selected based on ennemi criterion.
    """
    def _fit(self, X, y=None, **fit_params):
        vals = [ennemi.estimate_corr(np.ravel(y), X[:,i], preprocess=True) for i in range(X.shape[-1])]
        return np.array(vals).ravel()