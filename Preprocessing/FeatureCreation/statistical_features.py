from math import ceil

import scipy


def create_statistical_features(df, features_to_select=[],
                                statistical_features=['tmean', 'tstd', 'tmax', 'tmin', 'skew', 'moment', 'kurtosis'],
                                window_size=2, window_type="static"):
    df_slices = ceil(len(df) / window_size)

    for df_slice in range(0, df_slices):
        start = df_slice * window_size
        end = start + window_size
        for col in features_to_select:
            for idx, feat in enumerate(statistical_features):
                if window_type == "static":
                    if f'{col}_{feat}' in df.columns:
                        df[f'{col}_{feat}'][start:end] = getattr(scipy.stats, statistical_features[idx])(
                            df[col][start:end].to_numpy())
                    else:
                        df[f'{col}_{feat}'] = getattr(scipy.stats, statistical_features[idx])(
                            df[col][start:end].to_numpy())
                       # print("df[f'{col}_{feat}'] ", df[f'{col}_{feat}'])
                else:
                    # if feat == "tmax":
                    zscore = lambda x: getattr(scipy.stats, statistical_features[idx])(x)
                    # zscore = lambda x: scipy.stats.tmean(x)
                    # df[f'{col}_{feat}'] = df[col].rolling(2).apply(getattr(scipy.stats, statistical_features[idx]))
                    df[f'{col}_{feat}'] = df[col].rolling(window_size).apply(zscore).fillna(0)

                    # df[f'{col}_{feat}'] = df[col].rolling(window_size).mean()
            if 'tmin' in statistical_features and 'tmax' in statistical_features:
                df[f'{col}_ptop'] = df[f'{col}_tmax'] - df[f'{col}_tmin']
            if 'tmean' in statistical_features and 'tmax' in statistical_features:
                mean = df[f'{col}_tmean'].mask(df[f'{col}_tmean'] == 0).fillna(1.0)
                df[f'{col}_if'] = df[f'{col}_tmax'] // mean
    return df
