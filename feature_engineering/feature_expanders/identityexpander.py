from .feature_expansion import FeatureExpansion


class IdentityExpander(FeatureExpansion):
    """
    Feature Expansion - identity
    Base class for feature expansion transformers.
    """
    def __init__(self, features_to_transform=None, **kwargs):
        super().__init__(features_to_transform=features_to_transform)