import numpy as np
import pandas as pd

from .filter import Filter


class Envelope_MA(Filter):
    """
        Envelope detector through moving average (parameter T of moving average filter can be set)
    """
    _envelope_types = {'avg', 'min', 'max'}
    T = 10
    envelope_h_ = None
    envelope_l_ = None
    envelope_avg_ = None
    env_type = 'avg'

    def __init__(self, T=10, env_type='avg', remove_offset=False, keep_nans=False, **kwargs):
        super().__init__(remove_offset=remove_offset, keep_nans=keep_nans)
        self._set_attrs(T=T, env_type=env_type)

    def _transform(self, X):
        """
        Calculate envelope.
        This is taken from https://stackoverflow.com/a/69357933
        @param x: Input feature vector (n_samples, n_features)
        @param y: Target feature vector (n_samples)
        """
        X_df = pd.DataFrame(X) if not isinstance(X, (pd.DataFrame, pd.Series)) else X
        self.envelope_h_ = X_df.rolling(window=self.T).max().shift(int(-self.T / 2))
        self.envelope_l_ = X_df.rolling(window=self.T).min().shift(int(-self.T / 2))
        self.envelope_avg_ = np.mean(np.dstack((self.envelope_l_, self.envelope_h_)),axis=-1)
        return self.envelope_l_ if self.env_type == 'min' else self.envelope_h_ if self.env_type == 'max' else self.envelope_avg_

    def get_avg_env(self):
        """
        Get average envelope
        """
        return self.envelope_avg_

    def get_max_env(self):
        """
            Get upper bound of envelope
        """
        return self.envelope_h_

    def get_min_env(self):
        """
        Get lower bound of envelope
        """
        return self.envelope_l_

    def _get_feature_names_out(self, feature_names=None):
        return [f'{feat}_env_{self.env_type}_{self.T}' for feat in feature_names]