from .feature_expansion import FeatureExpansion


class IdentityExpander(FeatureExpansion):
    """
    Identity Expander
    This class does not expand the features.
    Instantiate this class if you do not want to use any expansion.
    """
    def __init__(self, **kwargs):
        pass

    def _fit(self, x=None, y=None):
        pass

    def _transform(self, x=None):
        return x

    def _get_feature_names(self, feature_names=None):
        return feature_names


