import numpy as np

from ModelTraining.feature_engineering.featureengineeringbasic.featureselectors.feature_selector import FeatureSelector


class SelectorByName(FeatureSelector):
    """
    Selector by name:
    Select features by name
    """
    selected_feat_names = None
    feature_names_in_ = None

    def __init__(self, feat_names=[], selected_feat_names=[], **kwargs):
        super().__init__(**kwargs)
        self.feature_names_in_ = np.array(feat_names)
        self.selected_feat_names = np.array(selected_feat_names)

    def _fit(self, X, y, **fit_params):
        return np.array([name in self.selected_feat_names for name in self.feature_names_in_])

    def _get_support_mask(self):
        return np.array([name in self.selected_feat_names for name in self.feature_names_in_])