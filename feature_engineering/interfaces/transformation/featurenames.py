

class FeatureNames:

    def get_feature_names_out(self, feature_names=None):
        """
        Get feature names
        @param feature_names: Input feature names
        @return: Expanded feature names
        """
        if feature_names is None:
            return None
        return self._get_feature_names_out(feature_names)

    def _get_feature_names_out(self, feature_names=None):
        """
        Get feature names - Override this method.
        @param feature_names: Input feature names
        @return: Expanded feature names
        """
        return feature_names

