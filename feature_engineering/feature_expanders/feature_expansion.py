from sklearn.base import TransformerMixin, BaseEstimator
from ..interfaces import PickleInterface, Reshape, BaseFitTransform, FeatureNames


class FeatureExpansion(TransformerMixin, PickleInterface, Reshape, BaseFitTransform, BaseEstimator, FeatureNames):
    """
    Feature Expansion
    Base class for feature expansion transformers.
    Implements scikit-learn's TransformerMixin interface, allows storing and loading from pickle
    """

    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer to samples. Calls self._fit
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        X = self.reshape_data(X)
        if X.shape[1] > 0:
            self._fit(X, y)
        return self

    def transform(self, X):
        """
        Transform features. Calls self._transform
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Transformed sample vector (n_samples, n_features_expanded)
        """
        # Reshape if necessary
        X_reshaped = self.reshape_data(X)
        return self._transform(X_reshaped)

