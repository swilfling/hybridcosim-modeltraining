from sklearn.base import TransformerMixin
from ..interfaces import PickleInterface, FeatureNames, BaseFitTransform, MaskFeats


class Transformer_inplace(TransformerMixin, PickleInterface, FeatureNames, BaseFitTransform, MaskFeats):
    """
    Transform only selected features - keep other features
    """
    def fit(self, X, y=None, **fit_params):
        self._fit(self.mask_feats(X), y, **fit_params)
        return self

    def transform(self, X):
        """
        Transform data
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @return: transformed features
        """
        x_transf = self._transform(self.mask_feats(X))
        return self.combine_feats(x_transf, X)

    def get_feature_names_out(self, feature_names=None):
        """
        Get output feature names
        @param feature_names: input feature names
        @return: transformed feature names
        """
        if feature_names is None:
            return None
        feat_names_basic = self.mask_feats(feature_names, inverse=True)
        feature_names_tr = self._get_feature_names_out(self.mask_feats(feature_names))
        return self.combine_feats(feature_names_tr, feat_names_basic)

