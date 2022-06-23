import pandas as pd
from sklearn.base import TransformerMixin

from ..interfaces.pickleinterface import PickleInterface


class Transformer_SelectedFeats(TransformerMixin, PickleInterface):
    """
    Transform only selected features - keep other features
    """
    features_to_transform = None

    def __init__(self, features_to_transform=None):
        self.features_to_transform = features_to_transform

    def fit(self, X, y=None, **fit_params):
        if self.features_to_transform is not None:
            if isinstance(X, pd.DataFrame):
                X_to_transf = X[self.features_to_transform]
            else:
                X_to_transf = X[:,self.features_to_transform]
        else:
            X_to_transf = X
        self._fit(X_to_transf, y, **fit_params)
        return self

    def transform(self, X):
        """
        Transform data
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @return: transformed features
        """
        if self.features_to_transform is not None:
             X_to_transf = X[self.features_to_transform] if isinstance(X, pd.DataFrame) else X[:, self.features_to_transform]
        else:
            X_to_transf = X
        x_transf = self._transform(X_to_transf)
        if self.features_to_transform is not None:
            x_transf_new = X.copy()
            if isinstance(X, pd.DataFrame):
                x_transf_new[self.features_to_transform] = x_transf
            else:
                x_transf_new[:, self.features_to_transform] = x_transf
            return x_transf_new
        return x_transf

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