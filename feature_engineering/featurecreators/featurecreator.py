from sklearn.base import TransformerMixin
from ..interfaces.pickleinterface import PickleInterface


class FeatureCreator(TransformerMixin, PickleInterface):
    """
    Basic feature creator
    """
    selected_feats = []

    def __init__(self, selected_feats=[], **kwargs):
        self.selected_feats = selected_feats
        pass

    def fit(self, X, y=None, **fit_params):
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

    def get_feature_names_out(self, feature_names=None):
        """
        Get output feature names
        @param feature_names: input feature names
        @return: output feature names
        """
        return (feature_names if feature_names is not None else []) + self.get_additional_feat_names()