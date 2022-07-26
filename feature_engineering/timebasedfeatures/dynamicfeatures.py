import numpy as np
from ...datamodels.datamodels.processing.shape import split_into_target_segments
from ..featureengineering.interfaces import BasicTransformer


class DynamicFeatures(BasicTransformer):
    """
    This class creates dynamic features with a certain lookback.
    Options:
        - flatten_dynamic_feats - all created features are added to the second dimension of the input data
        - return_3d_array: if flattening dynamic features, still a 3-d array can be returned through this option
    """
    lookback_horizon: int = 0
    flatten_dynamic_feats = False
    return_3d_array = False

    def __init__(self, lookback_horizon=5, flatten_dynamic_feats=False, return_3d_array=False, **kwargs):
        self.lookback_horizon = lookback_horizon
        self.flatten_dynamic_feats = flatten_dynamic_feats
        self.return_3d_array = return_3d_array

    def _transform(self, X):
        """
        Split into target segments
        @param X: input features (n_samples, n_features)
        @return: transformed features (n_samples, lookback + 1, n_features) or (n_samples, n_features)
        """
        X_transf, _ = split_into_target_segments(X, None, lookback_horizon=self.lookback_horizon, prediction_horizon=0)
        X_transf = np.concatenate((np.zeros((self.lookback_horizon, *X_transf.shape[1:])),X_transf))
        if self.flatten_dynamic_feats:
            X_transf = X_transf.reshape(X_transf.shape[0], -1)
        if self.return_3d_array:
            if isinstance(X_transf, np.ndarray):
                if X_transf.ndim == 2:
                    X_transf = X_transf.reshape(X_transf.shape[0], 1, X_transf.shape[1])
        return X_transf

    def get_additional_feat_names(self, feature_names=None):
        return [f'{name}_{lag}' for lag in np.flip(np.arange(1, self.lookback_horizon + 1)) for name in feature_names]

    def _get_feature_names_out(self, feature_names=None):
        return self.get_additional_feat_names(feature_names) + list(feature_names)
