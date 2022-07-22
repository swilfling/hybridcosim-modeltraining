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

