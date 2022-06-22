import numpy as np
import pandas as pd

from .filter import Filter


class Envelope_MA(Filter):
    """
        Envelope detector through moving average (parameter T of moving average filter can be set)
    """
    T = 10
    envelope_h = None
    envelope_l = None
    envelope_avg = None

    def __init__(self, T=10, **kwargs):
        super().__init__(**kwargs)
        self._set_attrs(T=T)

    def _fit(self, X, y=None, **fit_params):
        return None

    def _transform_feats(self, X):
        """
        Calculate envelope.
        This is taken from https://stackoverflow.com/a/69357933
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        X_df = pd.DataFrame(X)
        self.envelope_h = X_df.rolling(window=self.T).max().shift(int(-self.T / 2))
        self.envelope_l = X_df.rolling(window=self.T).min().shift(int(-self.T / 2))
        self.envelope_avg = np.mean(np.dstack((self.envelope_l, self.envelope_h)),axis=-1)
        return self.envelope_avg

    def get_max_env(self):
        """
            Get upper bound of envelope
        """
        return self.envelope_h

    def get_min_env(self):
        """
        Get lower bound of envelope
        """
        return self.envelope_l