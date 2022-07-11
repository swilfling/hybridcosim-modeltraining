from math import ceil
import pandas as pd
import scipy
from .featurecreator import FeatureCreator


class StatisticalFeatures(FeatureCreator):
    """
    Statistical features.
    Parameters: Windowsize, window type, features to select
    """
    _all_stat_feats = ['tmean', 'tstd', 'tmax', 'tmin', 'skew', 'moment', 'kurtosis']
    window_size = 2
    window_type = 'static'
    features_to_select = []
    statistical_features = []

    def __init__(self, features_to_transform=None, statistical_features=['tmin', 'tmax'], features_to_select=[], window_size=2, window_type='static', **kwargs):
        super().__init__(features_to_transform=features_to_transform)
        self.statistical_features = statistical_features
        self.window_type = window_type
        self.window_size = window_size
        self.features_to_select = features_to_select

    def _transform(self, X):
        """
        Add statistical features
        @param X: input data
        @return: transformed data
        """
        df = X if isinstance(X, (pd.DataFrame, pd.Series)) else pd.DataFrame(X)
        df_slices = ceil(len(df) / self.window_size)

        df_new = pd.DataFrame(index=df.index)
        for df_slice in range(0, df_slices):
            start = df_slice * self.window_size
            end = start + self.window_size
            for col in self.features_to_select:
                for idx, feat in enumerate(self.statistical_features):
                    if self.window_type == "static":
                        if f'{col}_{feat}' in df.columns:
                            df_new[f'{col}_{feat}_{self.window_size}'][start:end] = getattr(scipy.stats, self.statistical_features[idx])(
                                df[col][start:end].to_numpy())
                        else:
                            df_new[f'{col}_{feat}_{self.window_size}'] = getattr(scipy.stats, self.statistical_features[idx])(
                                df[col][start:end].to_numpy())
                           # print("df[f'{col}_{feat}'] ", df[f'{col}_{feat}'])
                    else:
                        # if feat == "tmax":
                        zscore = lambda x: getattr(scipy.stats, self.statistical_features[idx])(x)
                        # zscore = lambda x: scipy.stats.tmean(x)
                        # df[f'{col}_{feat}'] = df[col].rolling(2).apply(getattr(scipy.stats, statistical_features[idx]))
                        df_new[f'{col}_{feat}_{self.window_size}'] = df[col].rolling(self.window_size).apply(zscore).fillna(0)

                        # df[f'{col}_{feat}'] = df[col].rolling(window_size).mean()
                if 'tmin' in self.statistical_features and 'tmax' in self.statistical_features and 'ptop' in self.statistical_features:
                    df_new[f'{col}_ptop_{self.window_size}'] = df_new[f'{col}_tmax_{self.window_size}'] - df[f'{col}_tmin_{self.window_size}']
                if 'tmean' in self.statistical_features and 'tmax' in self.statistical_features:
                    mean = df_new[f'{col}_tmean_{self.window_size}'].mask(df_new[f'{col}_tmean_{self.window_size}'] == 0).fillna(1.0)
                    if 'if' in self.statistical_features:
                        df_new[f'{col}_if_{self.window_size}'] = df_new[f'{col}_tmax_{self.window_size}'] // mean
        df = df.join(df_new)
        df = df.fillna(0)
        return df if isinstance(X, (pd.Series, pd.DataFrame)) else df.values

    def get_additional_feat_names(self):
        return [f"{name}_{stat_name}_{self.window_size}" for name in self.features_to_select for stat_name in self.statistical_features]