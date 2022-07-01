from . import FeatureCreator
from ..interfaces import BaseFitTransform
import numpy as np
import pandas as pd
from ModelTraining.datamodels.datamodels.processing.shape import split_into_target_segments
from sklearn.base import BaseEstimator, TransformerMixin


class SampleCut(BaseEstimator, TransformerMixin, BaseFitTransform):
    num_samples = 0
    internal_samples_ = None

    def __init__(self, num_samples=0, **kwargs):
        self.num_samples = num_samples

    def transform(self, X):
        self.internal_samples_ = X[:self.num_samples]
        return X[self.num_samples:]
    
    def inverse_transform(self, X):
        #return X[self.num_samples:]
        return np.concatenate((self.internal_samples_, X))


class SampleReInit(BaseEstimator, TransformerMixin, BaseFitTransform):
    num_samples = 0
    init_val = 0
    input_shape_= None

    def __init__(self, num_samples=0, init_val=0, **kwargs):
        self.num_samples = num_samples
        self.init_val = init_val

    def _fit(self, X, y, **fit_params):
        self.input_shape_ = X.shape

    def transform(self, X):
        X[self.num_samples:] = self.init_val
        return X

    def inverse_transform(self, X):
        # return X[self.num_samples:]
        return X

    def create_lookback_samples(self):
        return np.ones((self.num_samples, *self.input_shape_[1:])) * self.init_val if self.input_shape_ is not None else None


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

    def __init__(self, lookback_horizon=5, flatten_dynamic_feats=False, **kwargs):
        super().__init__(**kwargs)
        self.lookback_horizon = lookback_horizon
        self.flatten_dynamic_feats = flatten_dynamic_feats

    def _transform(self, X):
        """
        Split into target segments
        @param X: input features (n_samples, n_features)
        @return: transformed features (n_samples, lookback + 1, n_features) or (n_samples, n_features)
        """
        X_transf, _ = split_into_target_segments(X, None, lookback_horizon=self.lookback_horizon, prediction_horizon=0)
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
        if (isinstance(X_orig, np.ndarray) and X_orig.ndim > 1) or isinstance(X_orig, pd.DataFrame):
            x_orig_init = SampleCut(self.lookback_horizon).transform(x_orig_init)
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
