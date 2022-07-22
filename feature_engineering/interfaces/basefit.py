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
