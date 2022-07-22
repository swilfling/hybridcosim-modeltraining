import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from ModelTraining.feature_engineering.interfaces import PickleInterface, Reshape, BaseFit


class FeatureSelector(SelectorMixin, BaseFit, BaseEstimator, PickleInterface, Reshape):
    """
        FeatureSelector - implements SelectorMixin interface, can be stored to pickle.
        Basic implementation: threshold
        Options:
            - omit_zero_samples: Omit zero samples from selection
    """
    n_features_in_ = 0

    def __init__(self, **kwargs):
        pass

    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer - Overrides TransformerMixin method.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        X = self.reshape_x(X)
        y = self.reshape_y(y)
        self.n_features_in_ = X.shape[1]
        return super().fit(X, y, **fit_params)

    def transform(self, X):
        """
        Transform samples.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Output feature vector (n_samples, n_features) or (n_samples, n_selected_features * lookback)
        """
        x_tr = super().transform(self.reshape_x(X))
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(data=x_tr, columns=self.get_feature_names_out(X.columns))
        return x_tr

    def get_num_selected_features(self):
        """
        Get number of selected features - override if necessary.
        """
        return np.sum(self._get_support_mask())

    def _get_support_mask(self):
        """
        Get mask for feature selection
        """
        return np.ones(self.n_features_in_)

    def get_num_features(self):
        """
        Get total number of features - override if necessary.
        """
        return self._get_support_mask().shape[0]

    def get_metrics(self):
        """
        return selected and total features
        """
        return {'selected_features': self.get_num_selected_features(), 'all_features': self.get_num_features()}

    def print_metrics(self):
        """
        Print number of selected and total features
        """
        print(f'Selecting features: {self.get_num_selected_features()} of {self.get_num_features()}')