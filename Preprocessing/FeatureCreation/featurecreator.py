from sklearn.base import TransformerMixin
from ...datamodels.datamodels.wrappers.feature_extension import StoreInterface


class FeatureCreator(TransformerMixin, StoreInterface):
    """
    Basic feature creator
    """
    selected_feats = []

    def __init__(self, selected_feats=[], **kwargs):
        self.selected_feats = selected_feats
        pass

    def fit(self, X, y, **fit_params):
        """
        Fit to data. This function is not used here
        """
        return self

    def transform(self, X):
        """
        Transform - add additional features
         @param X: input data
         @return: transformed data
         """
        return X

    def get_additional_feat_names(self):
        """
        Get additional feature names
        @param X: input data
        @return: transformed data
        """
        return []
