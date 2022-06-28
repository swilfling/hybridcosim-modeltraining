from . import FeatureCreator
import numpy as np
import pandas as pd
from ModelTraining.datamodels.datamodels.processing.shape import split_into_target_segments
from sklearn.base import BaseEstimator


class DynamicFeatures(FeatureCreator, BaseEstimator):
    """
    This class creates dynamic features with a certain lookback.
    Options:
        - flatten_dynamic_feats - all created features are added to the second dimension of the input data
        - return_3d_array: if flattening dynamic features, still a 3-d array can be returned through this option
    """
    lookback_horizon: int = 0
    flatten_dynamic_feats = False
    return_3d_arrray = False
    init_val = 0

    def __init__(self, lookback_horizon=5, flatten_dynamic_feats=False, init_val=0, **kwargs):
        super().__init__(**kwargs)
        self.lookback_horizon = lookback_horizon
        self.init_val = init_val
        self.flatten_dynamic_feats = flatten_dynamic_feats

    def _fit(self, X, y=None, **fit_params):
        # Remove samples of size self.lookback_horizon
        if y is not None:
            if X.shape[0] == y.shape[0]:
                if isinstance(y, np.ndarray):
                    #np.delete(y, np.arange(self.lookback_horizon), axis=0)
                    y[:self.lookback_horizon] = self.init_val
                    #y[:-self.lookback_horizon] = y[self.lookback_horizon:]
                if isinstance(y, pd.Series) or isinstance(y, pd.DataFrame):
                    #y.drop(index=y.index[range(self.lookback_horizon)], inplace=True)
                    y[:self.lookback_horizon] = self.init_val

    def _transform(self, X):
        """
        Split into target segments
        @param X: input features (n_samples, n_features)
        @return: transformed features (n_samples, lookback + 1, n_features) or (n_samples, n_features)
        """
        X_transf, _ = split_into_target_segments(X, None, lookback_horizon=self.lookback_horizon, prediction_horizon=0)
        init = np.ones((self.lookback_horizon, *X_transf.shape[1:])) * self.init_val
        X_transf = np.concatenate((init, X_transf))
        if self.flatten_dynamic_feats:
            X_transf = X_transf.reshape(X_transf.shape[0], -1)
        return X_transf

    def combine_feats(self, X_transf, X_orig):
        """
        Combine transformed and original features
        @param X_transf: array of transformed feats
        @param X_orig: original feature vector
        @return: full array
        """
        x_orig_init = X_orig.copy()
        x_orig_init[:self.lookback_horizon] = self.init_val
        feats = super(DynamicFeatures, self).combine_feats(X_transf, x_orig_init)
        if self.return_3d_arrray:
            if isinstance(feats, np.ndarray):
                if feats.ndim == 2:
                    feats = feats.reshape(feats.shape[0], 1, feats.shape[1])
        return feats

    def get_additional_feat_names(self, feature_names=None):
        return [f'{name}_{lag}' for lag in np.flip(np.arange(1, self.lookback_horizon + 1)) for name in feature_names]

    def _get_feature_names_out(self, feature_names=None):
        return self.get_additional_feat_names(feature_names) + list(feature_names)
