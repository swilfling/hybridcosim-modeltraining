import numpy as np

from .feature_selector import FeatureSelector


class IdentitySelector(FeatureSelector):
    """
    Identity:
    All features are selected.
    """
    def _get_support_mask(self):
        return np.array([True] * self.n_features_in_)