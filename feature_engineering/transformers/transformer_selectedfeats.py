import pandas as pd
from sklearn.base import TransformerMixin
import numpy as np

from ..interfaces.pickleinterface import PickleInterface


class Transformer_SelectedFeats(TransformerMixin, PickleInterface):
    """
    Transform only selected features - keep other features
    """
    features_to_transform = None

    def __init__(self, features_to_transform=None):
        self.features_to_transform = features_to_transform

    def fit(self, X, y=None, **fit_params):
        self._fit(self.mask_feats(X), y, **fit_params)
        return self

    def mask_feats(self, X, inverse=False):
        """
        Select features to transform
        @param X: all features
        @return: selected features
        """
        mask = np.bitwise_not(self.features_to_transform) if inverse else self.features_to_transform
        if self.features_to_transform is not None:
            if isinstance(X, pd.DataFrame):
                return X[X.columns[mask]]
            elif isinstance(X, np.ndarray):
                return X[..., self.features_to_transform]
            else:
                return np.array(X)[self.features_to_transform]

        return X

    def combine_feats(self, X_transf, X_orig):
        """
        Combine transformed and original features
        @param X_transf: array of transformed feats
        @param X_orig: original feature vector
        @return: full array
        """
        if self.features_to_transform is not None:
            # If transformation did not create new features, replace original by transformed values
            x_transf_new = X_orig.copy()
            if isinstance(X_orig, pd.DataFrame):
                x_transf_new[x_transf_new.columns[self.features_to_transform]] = X_transf
            else:
                x_transf_new[..., self.features_to_transform] = X_transf
            return x_transf_new

        else:
            return X_transf

    def transform(self, X):
        """
        Transform data
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @return: transformed features
        """
        x_transf = self._transform(self.mask_feats(X))
        return self.combine_feats(x_transf, X)

    def _transform(self, X):
        """
        Transformation method - Override this method
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @return: transformed features
        """
        return X

    def _fit(self, X, y=None, **fit_params):
        """
        Fitting method - Override this method
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        """
        pass

    def get_feature_names_out(self, feature_names=None):
        """
        Get output feature names
        @param feature_names: input feature names
        @return: transformed feature names
        """
        if feature_names is None:
            return None
        feat_names_to_transform = self.mask_feats(feature_names)
        feature_names_tr = self._get_feature_names_out(feat_names_to_transform)
        if self.features_to_transform is None:
            return feature_names_tr
        else:
            # number of features did not increase: replace names
            feat_names_out = np.array(feature_names)
            feat_names_out[self.features_to_transform] = feature_names_tr #todo fix this
            return feat_names_out

    def _get_feature_names_out(self, feature_names=None):
        return feature_names