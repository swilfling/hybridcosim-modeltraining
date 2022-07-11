import numpy as np


class Reshape:

    def reshape_data(self, X: np.ndarray):
        """
        Reshape data if 3-dimensional.
        @param X: data
        @return: reshaped data
        """
        if X.ndim == 3:
            return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        return X


class FitTransform:
    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer - Overrides TransformerMixin method.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        return self

    def transform(self, X):
        """
        Transform samples.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Output feature vector (n_samples, n_features) or (n_samples, n_selected_features * lookback)
        """
        return X


class BaseTransform:

    def transform(self, X):
        """
        Transform samples.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Output feature vector (n_samples, n_features) or (n_samples, n_selected_features * lookback)
        """
        return self._transform(X)

    def _transform(self, X):
        """
        Transformation method - Override this method
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @return: transformed features
        """
        return X


class BaseFit:

    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer - Overrides TransformerMixin method.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        self._fit(X, y, **fit_params)
        return self

    def _fit(self, X, y, **fit_params):
        """
        Fit transformer - Override this method!
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        pass


class BaseFitTransform(BaseFit, BaseTransform):
    pass