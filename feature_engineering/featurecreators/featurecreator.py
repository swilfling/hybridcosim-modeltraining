import pandas as pd
import numpy as np
from ..interfaces import PickleInterface, MaskFeats, FeatureNames, BaseFitTransform


class FeatureCreator(PickleInterface, MaskFeats, FeatureNames, BaseFitTransform):
    """
    Basic feature creator
    """
    selected_feats=None

    def __init__(self, selected_feats=[], **kwargs):
        super(FeatureCreator, self).__init__(**kwargs)
        self.selected_feats = selected_feats

    def get_additional_feat_names(self, feature_names=None):
        """
        Get additional feature names
        @param X: input data
        @return: transformed data
        """
        return []

    def combine_feats(self, X_transf, X_orig):
        if self.features_to_transform is not None:
            # If transformation created new features: concatenate basic and new features
            if isinstance(X_orig, pd.DataFrame):
                x_basic = self.mask_feats(X_orig, inverse=True)
                for i, name in enumerate(self._get_feature_names_out(self.mask_feats(X_orig.columns))):
                    x_basic[name] = X_transf[..., i]
                return x_basic
            else:
                x_basic = X_orig[np.bitwise_not(self.features_to_transform)]
                return np.concatenate((x_basic, X_transf), axis=-1)
        else:
            return X_transf

    def _get_feature_names_out(self, feature_names=None):
        """
        Get output feature names
        @param feature_names: input feature names
        @return: output feature names
        """
        return (feature_names if feature_names is not None else []) + self.get_additional_feat_names(feature_names)