from ..interfaces import BasicTransformer


class FeatureCreator(BasicTransformer):
    """
    Basic feature creator
    """
    def get_additional_feat_names(self):
        """
        Get additional feature names
        @return: list of feature names
        """
        return []

    def _get_feature_names_out(self, feature_names=None):
        return list(feature_names) + list(self.get_additional_feat_names())