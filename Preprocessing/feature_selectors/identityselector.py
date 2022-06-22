import numpy as np

from .feature_selector import FeatureSelector


class IdentitySelector(FeatureSelector):
    """
    Identity:
    All features are selected.
    """
    def _fit(self, X, y, **fit_params):
        return np.ones(X.shape[-1])

    def _get_support_mask(self):
        return np.array([True] * self.coef_.shape[-1])