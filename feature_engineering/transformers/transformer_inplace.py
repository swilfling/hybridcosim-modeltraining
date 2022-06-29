from sklearn.base import TransformerMixin
from ..interfaces import PickleInterface, BaseFitTransform, MaskFeats


class Transformer_inplace(TransformerMixin, PickleInterface, BaseFitTransform, MaskFeats):
    """
    Transform only selected features - keep other features
    """
    def __init__(self, **kwargs):
        MaskFeats.__init__(**kwargs)

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

