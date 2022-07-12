from imblearn.base import SamplerMixin
from sklearn.base import BaseEstimator

from ..resamplers import SampleCut_imblearn
from .transformer_maskfeats import Transformer_MaskFeats


class DynamicFeaturesSampleCut(SamplerMixin, BaseEstimator):
    lookback_horizon: int = 0
    flatten_dynamic_feats = True
    return_3d_array = False
    features_to_transform = None
    dyn_feats_ = None
    sample_cut_ = None

    def __init__(self, features_to_transform=None, lookback_horizon=5, flatten_dynamic_feats=False, return_3d_array=False, **kwargs):
        self.lookback_horizon = lookback_horizon
        self.flatten_dynamic_feats = flatten_dynamic_feats
        self.return_3d_array = return_3d_array
        self.features_to_transform = features_to_transform

    def fit_resample(self, X, y):
        # Omit checks for number of features
        return self._fit_resample(X, y)

    def _fit_resample(self, X, y):
        return self.fit(X, y).resample(X, y)

    def fit(self, X, y):
        self.dyn_feats_ = Transformer_MaskFeats(transformer_type='DynamicFeatures',
                                                transformer_params={'lookback_horizon':self.lookback_horizon,
                                                                    'flatten_dynamic_feats':self.flatten_dynamic_feats,
                                                                    'return_3d_array':self.return_3d_array},
                                                features_to_transform=self.features_to_transform,
                                                mask_type='MaskFeats_Expanded')
        self.sample_cut_ = SampleCut_imblearn(self.lookback_horizon)
        self.dyn_feats_.fit(X, y)
        self.sample_cut_.fit(X, y)
        return self

    def resample(self, X, y):
        if y is None:
            return self.sample_cut_._fit_resample(self.dyn_feats_.fit_transform(X, y), y)
        else:
            return self.sample_cut_.fit_resample(self.dyn_feats_.fit_transform(X, y), y)

    def get_feature_names_out(self, feature_names=None):
        return self.dyn_feats_.get_feature_names_out(feature_names)