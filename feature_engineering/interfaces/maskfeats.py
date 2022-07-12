import numpy as np
import pandas as pd
from . import BasicInterface


class FeatureNames:

    def get_feature_names_out(self, feature_names=None):
        """
        Get feature names
        @param feature_names: Input feature names
        @return: Expanded feature names
        """
        if feature_names is None:
            return None
        return self._get_feature_names_out(feature_names)

    def _get_feature_names_out(self, feature_names=None):
        """
        Get feature names - Override this method.
        @param feature_names: Input feature names
        @return: Expanded feature names
        """
        return feature_names


class MaskFeats(FeatureNames, BasicInterface):
    features_to_transform = None

    def __init__(self, features_to_transform=None):
        self.features_to_transform = features_to_transform

    def mask_feats(self, X, inverse=False):
        """
        Select features to transform
        @param X: all features
        @param inverse: invert features_to_transform
        @return: selected features
        """
        if self.features_to_transform is not None:
            mask = np.bitwise_not(self.features_to_transform) if inverse else np.array(self.features_to_transform)
            if isinstance(X, pd.DataFrame):
                return X[X.columns[mask]]
            elif isinstance(X, np.ndarray):
                return X[..., mask]
            else:
                return np.array(X)[mask]
        else:
            return None if inverse else X

    def combine_feats(self, X_transf, X_orig, feature_names=None):
        return X_transf

    def get_feature_names_out(self, feature_names=None):
        """
        Get output feature names
        @param feature_names: input feature names
        @return: transformed feature names
        """
        if feature_names is None:
            return None
        feat_names_to_transform = self.mask_feats(feature_names)
        feature_names_tr = self._get_feature_names_out(feat_names_to_transform)
        return self.combine_feats(np.array(feature_names_tr), feature_names)


class MaskFeats_Inplace(MaskFeats):

    def __init__(self, features_to_transform=None):
        super().__init__(features_to_transform=features_to_transform)

    def combine_feats(self, X_transf, X_orig, feature_names=None):
        """
        Combine transformed and original features
        @param X_transf: array of transformed feats
        @param X_orig: original feature vector
        @return: full array
        """
        if self.features_to_transform is not None:
            # If transformation did not create new features, replace original by transformed values
            x_transf_new = X_orig.copy()
            if isinstance(X_orig, pd.DataFrame):
                x_transf_new[x_transf_new.columns[self.features_to_transform]] = X_transf
                if feature_names is not None:
                    column_names = np.array(x_transf_new.columns)
                    column_names[self.features_to_transform] = feature_names
                    x_transf_new.columns = column_names
            elif isinstance(X_orig, np.ndarray):
                x_transf_new[..., self.features_to_transform] = X_transf
            else:
                x_transf_new = np.array(x_transf_new)
                x_transf_new[self.features_to_transform] = X_transf
            return x_transf_new

        else:
            return X_transf


class MaskFeats_Addition(MaskFeats):

    def __init__(self, features_to_transform=None):
        super().__init__(features_to_transform=features_to_transform)

    def combine_feats(self, X_transf, X_orig, feature_names=None):
        """
        Combine transformed and original features
        @param X_transf: array of transformed feats
        @param X_orig: original feature vector
        @return: full array
        """
        if self.features_to_transform is not None:
            # If transformation did not create new features, replace original by transformed values
            x_transf_new = X_orig.copy()
            if isinstance(X_orig, pd.DataFrame):
                dummy_feat_names = [f'feat_{i}' for i in range(X_orig.shape[-1], X_orig.shape[-1] + X_transf.shape[-1])]
                feat_names_new = dummy_feat_names if feature_names is None else feature_names
                x_transf_new[feat_names_new] = X_transf
            else:
                x_transf_new = np.concatenate((np.array(X_orig), np.array(X_transf)))
            return x_transf_new
        else:
            return X_transf


class MaskFeats_Expanded(MaskFeats):

    def __init__(self, features_to_transform=None):
        super().__init__(features_to_transform=features_to_transform)

    def combine_feats(self, X_transf, X_orig, feature_names=None):
        if self.features_to_transform is not None:
            # If transformation created new features: concatenate basic and new features
            if isinstance(X_orig, pd.DataFrame):
                x_basic = self.mask_feats(X_orig, inverse=True)
                dummy_feat_names = [f'feat_{i}' for i in range(x_basic.shape[-1], x_basic.shape[-1] + X_transf.shape[-1])]
                feat_names = dummy_feat_names if feature_names is None else feature_names
                for i, name in enumerate(feat_names):
                    x_basic[name] = X_transf[X_transf.columns[i]] if isinstance(X_transf, pd.DataFrame) else X_transf[..., i]
                return x_basic
            else:
                x_basic = self.mask_feats(X_orig, inverse=True)
                return np.concatenate((x_basic, X_transf), axis=-1)
        else:
            return X_transf
