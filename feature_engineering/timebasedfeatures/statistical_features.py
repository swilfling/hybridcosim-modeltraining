from math import ceil
import pandas as pd
import scipy.stats
import numpy as np
from ..interfaces import BasicTransformer


class StatisticalFeatures(BasicTransformer):
    """
    Statistical features.
    Parameters: Windowsize, window type, features to select
    """
    _all_stat_feats = ['tmean', 'tstd', 'tmax', 'tmin', 'skew', 'moment', 'kurtosis']
    window_size = 2
    window_type = 'static'
    statistical_features = []

    def __init__(self, statistical_features=['tmin', 'tmax'], window_size=2, window_type='static', **kwargs):
        self.statistical_features = statistical_features
        self.window_type = window_type
        self.window_size = window_size

    def _transform(self, X):
        """
        Add statistical features
        @param X: input data
        @return: transformed data
        """
        df = X if isinstance(X, (pd.DataFrame)) else pd.DataFrame(X)
        df_slices = ceil(len(df) / self.window_size)

        zscores = [getattr(scipy.stats, func) for func in self.statistical_features]
        df_new = pd.DataFrame(index=df.index)
        group_range = np.repeat(np.arange(df.shape[0] // self.window_size),  self.window_size)
        df_for_group = df[:len(group_range)]
        if self.window_type == "static":
            for col in df_for_group.columns:
                #x_vals = df_for_group[col].values
                #array_splits = np.array_split(x_vals, x_vals.shape[0] / self.window_size)
                for idx, feat in enumerate(self.statistical_features):
                    x_vals_tr = df_for_group[col].groupby(group_range).apply(zscores[idx])
                    #x_vals_tr = np.array([zscores[idx](split, axis=0) for split in array_splits])
                    x_vals_tr = np.repeat(x_vals_tr.values, self.window_size, axis=0)
                    #df_cols = [f'{col}_{feat}_{self.window_size}' for col in df.columns]
                    #df_vals = pd.DataFrame(data=x_vals_tr, columns=df_cols, index=df.index)
                    df_new = df_new.join(pd.DataFrame(index=df_for_group.index, data=x_vals_tr,columns=[col]).add_suffix(f"_{feat}_{self.window_size}")).fillna(0)
        else:
            df = X if isinstance(X, (pd.DataFrame)) else pd.DataFrame(X)
            for col in df.columns:
                for idx, feat in enumerate(self.statistical_features):
                    # if feat == "tmax":
                    zscore = lambda x: getattr(scipy.stats, self.statistical_features[idx])(x)
                    # zscore = lambda x: scipy.stats.tmean(x)
                    # df[f'{col}_{feat}'] = df[col].rolling(2).apply(getattr(scipy.stats, statistical_features[idx]))
                    df_new[f'{col}_{feat}_{self.window_size}'] = df[col].rolling(self.window_size).apply(zscore).fillna(0)

                        # df[f'{col}_{feat}'] = df[col].rolling(window_size).mean()
        # Features that require others
        for col in df.columns:
            if 'tmin' in self.statistical_features and 'tmax' in self.statistical_features and 'ptop' in self.statistical_features:
                df_new[f'{col}_ptop_{self.window_size}'] = df_new[f'{col}_tmax_{self.window_size}'] - df_new[f'{col}_tmin_{self.window_size}']
            if 'tmean' in self.statistical_features and 'tmax' in self.statistical_features:
                if 'if' in self.statistical_features:
                    mean = df_new[f'{col}_tmean_{self.window_size}'].mask(
                        df_new[f'{col}_tmean_{self.window_size}'] == 0).fillna(1.0)
                    df_new[f'{col}_if_{self.window_size}'] = df_new[f'{col}_tmax_{self.window_size}'] // mean
        df_new = df_new.fillna(0)
        return df_new if isinstance(X, (pd.Series, pd.DataFrame)) else df_new.values

    def _get_feature_names_out(self, feature_names=None):
        return [f"{name}_{stat_name}_{self.window_size}" for name in feature_names for stat_name in self.statistical_features]