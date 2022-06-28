import numpy as np
import pandas as pd


class Reshape:

    def reshape_data(self, X: np.ndarray):
        """
        Reshape data if 3-dimensional.
        @param X: data
        @return: reshaped data
        """
        if X.ndim == 3:
            return X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        return X


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


class Transformer:
    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer - Overrides TransformerMixin method.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        return self

    def transform(self, X):
        """
        Transform samples.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Output feature vector (n_samples, n_features) or (n_samples, n_selected_features * lookback)
        """
        return X


class BaseTransform:

    def transform(self, X):
        """
        Transform samples.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Output feature vector (n_samples, n_features) or (n_samples, n_selected_features * lookback)
        """
        return X

    def _transform(self, X):
        """
        Transformation method - Override this method
        @param X: Input feature vector (n_samples, n_features) - supports pd dataframe
        @return: transformed features
        """
        return X



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


class BaseFitTransform(BaseFit, BaseTransform):

    def __init__(self, **kwargs):
        pass

class MaskFeats:
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
        mask = np.bitwise_not(self.features_to_transform) if inverse else self.features_to_transform
        if self.features_to_transform is not None:
            if isinstance(X, pd.DataFrame):
                return X[X.columns[mask]]
            elif isinstance(X, np.ndarray):
                return X[..., self.features_to_transform]
            else:
                return np.array(X)[self.features_to_transform]
        return X

    def combine_feats(self, X_transf, X_orig):
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
            elif isinstance(X_orig, np.ndarray):
                x_transf_new[..., self.features_to_transform] = X_transf
            else:
                np.array(X_orig)[self.features_to_transform] = X_transf
            return x_transf_new

        else:
            return X_transf
