from . import FeatureExpansion


class IdentityExpander(FeatureExpansion):
    """
    Feature Expansion - identity
    Base class for feature expansion transformers.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)