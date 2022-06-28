from abc import abstractmethod
from sklearn.base import TransformerMixin
from ..interfaces import PickleInterface


class FeatureExpansion(TransformerMixin, PickleInterface):
    """
    Feature Expansion
    Base class for feature expansion transformers.
    Implements scikit-learn's TransformerMixin interface, allows storing and loading from pickle
    """
    def get_feature_names_out(self, feature_names=None):
        """
        Get feature names
        @param feature_names: Input feature names
        @return: Expanded feature names
        """
        if feature_names is None:
            return None
        return self._get_feature_names(feature_names)

    def reshape_data(self, X):
        """
        Reshape data if 3-dimensional.
        """
        if X.ndim == 3:
            return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        return X

    def fit(self, X, y=None):
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
        @return: Transformed sample vector (n_samples, n_features_expanded) or (n_samples, lookback, n_features_expanded)
        """
        # Reshape if necessary
        x_expanded = self._transform(self.reshape_data(X))
        # Reshape to 3D if necessary
        x_expanded = x_expanded.reshape((X.shape[0], X.shape[1], int(x_expanded.shape[1] / X.shape[1]))) if X.ndim == 3 else x_expanded
        return x_expanded

    ################################################## Internal methods - override these ###############################

    @abstractmethod
    def _get_feature_names(self, feature_names=None):
        """
        Get feature names - Override this method.
        @param feature_names: Input feature names
        @return: Expanded feature names
        """
        raise NotImplementedError()

    @abstractmethod
    def _fit(self, X, y=None):
        """
        Fit transformer to samples - override this method.
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        raise NotImplementedError()

    @abstractmethod
    def _transform(self, X):
        """
        Transform features - Override this method.
        @param x: Input feature vector (n_samples, n_features)
        @return: Transformed sample vector (n_samples, n_features_expanded)
        """
        raise NotImplementedError()

