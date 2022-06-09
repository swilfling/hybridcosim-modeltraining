import numpy as np

from .feature_selector import FeatureSelector


class SelectorByName(FeatureSelector):
    """
    Selector by name:
    Select features by name
    """
    def __init__(self, feat_names=[], selected_feat_names=[], **kwargs):
        super().__init__(**kwargs)
        self.feature_names_in_ = feat_names
        self.selected_feat_names = selected_feat_names

    def _fit(self, X, y, **fit_params):
        return np.array([name in self.selected_feat_names for name in self.feature_names_in_])

    def _get_support_mask(self):
        return [name in self.selected_feat_names for name in self.feature_names_in_]