from sklearn.base import TransformerMixin
from ..interfaces import PickleInterface, BaseFitTransform, MaskFeats, BasicInterface


class Transformer_inplace(TransformerMixin, PickleInterface, BaseFitTransform, MaskFeats):
    """
    Transform only selected features - keep other features
    Transformed features replace basic features
    """
    transformer_type = ""
    transformer_params = {}
    transformer_ = None

    def __init__(self, transformer_type="", transformer_params={}, **kwargs):
        self.transformer_type = transformer_type
        self.transformer_params = transformer_params
        MaskFeats.__init__(self, **kwargs)

    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer.
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @param y: Target feature vector - optional
        @return: self
        """
        self.transformer_ = BasicInterface.from_name(self.transformer_type, **self.transformer_params)
        if self.transformer_ is not None:
            self.transformer_.fit(self.mask_feats(X), y, **fit_params)
        return self

    def transform(self, X):
        """
        Transform data
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @return: transformed features
        """
        x_transf = self.transformer_.transform(self.mask_feats(X))
        return self.combine_feats(x_transf, X)

    def _get_feature_names_out(self, feature_names=None):
        return self.transformer_.get_feature_names_out(feature_names) if self.transformer_ is not None else None