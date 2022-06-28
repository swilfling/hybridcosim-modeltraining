from ..interfaces import PickleInterface, MaskFeatsExpanded, BaseFitTransform


class FeatureCreator(PickleInterface, MaskFeatsExpanded, BaseFitTransform):
    """
    Basic feature creator
    """
    selected_feats=None

    def __init__(self, selected_feats=[], **kwargs):
        super(FeatureCreator, self).__init__(**kwargs)
        self.selected_feats = selected_feats

    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer to samples. Calls self._fit
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        self._fit(self.mask_feats(X), y)
        return self

    def transform(self, X):
        """
        Transform features. Calls self._transform
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Transformed sample vector (n_samples, n_features_expanded)
        """
        # Reshape if necessary
        return self.combine_feats(self._transform(self.mask_feats(X)), X)

    def get_additional_feat_names(self):
        return []

    def _get_feature_names_out(self, feature_names=None):
        return feature_names + self.get_additional_feat_names()