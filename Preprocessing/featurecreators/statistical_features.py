from math import ceil

import scipy
from .featurecreator import FeatureCreator


class StatisticalFeatures(FeatureCreator):
    """
    Statistical features.
    Parameters: Windowsize, window type, features to select
    """
    all_stat_feats = ['tmean', 'tstd', 'tmax', 'tmin', 'skew', 'moment', 'kurtosis']
    window_size = 2
    window_type = 'static'
    features_to_select = []
    statistical_features = []

    def __init__(self, statistical_features=['tmin', 'tmax'], features_to_select=[], window_size=2, window_type='static', **kwargs):
        super().__init__(**kwargs)
        self.statistical_features = statistical_features
        self.window_type = window_type
        self.window_size = window_size
        self.features_to_select = features_to_select

    def transform(self, X):
        """
        Add statistical features
        @param X: input data
        @return: transformed data
        """
        df = X
        df_slices = ceil(len(df) / self.window_size)

        for df_slice in range(0, df_slices):
            start = df_slice * self.window_size
            end = start + self.window_size
            for col in self.features_to_select:
                for idx, feat in enumerate(self.statistical_features):
                    if self.window_type == "static":
                        if f'{col}_{feat}' in df.columns:
                            df[f'{col}_{feat}'][start:end] = getattr(scipy.stats, self.statistical_features[idx])(
                                df[col][start:end].to_numpy())
                        else:
                            df[f'{col}_{feat}'] = getattr(scipy.stats, self.statistical_features[idx])(
                                df[col][start:end].to_numpy())
                           # print("df[f'{col}_{feat}'] ", df[f'{col}_{feat}'])
                    else:
                        # if feat == "tmax":
                        zscore = lambda x: getattr(scipy.stats, self.statistical_features[idx])(x)
                        # zscore = lambda x: scipy.stats.tmean(x)
                        # df[f'{col}_{feat}'] = df[col].rolling(2).apply(getattr(scipy.stats, statistical_features[idx]))
                        df[f'{col}_{feat}'] = df[col].rolling(self.window_size).apply(zscore).fillna(0)

                        # df[f'{col}_{feat}'] = df[col].rolling(window_size).mean()
                if 'tmin' in self.statistical_features and 'tmax' in self.statistical_features:
                    df[f'{col}_ptop'] = df[f'{col}_tmax'] - df[f'{col}_tmin']
                if 'tmean' in self.statistical_features and 'tmax' in self.statistical_features:
                    mean = df[f'{col}_tmean'].mask(df[f'{col}_tmean'] == 0).fillna(1.0)
                    df[f'{col}_if'] = df[f'{col}_tmax'] // mean
        return df

    def get_additional_feat_names(self):
        return [f"{name}_{stat_name}" for name in self.features_to_select for stat_name in self.statistical_features]