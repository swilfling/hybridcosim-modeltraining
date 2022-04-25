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
            df[f'{col}_ptop'] = df[f'{col}_tmax'] - df[f'{col}_tmin']
            mean = df[f'{col}_tmean'].mask(df[f'{col}_tmean'] == 0).fillna(1.0)
            df[f'{col}_if'] = df[f'{col}_tmax'] // mean
    return df


def add_stat_feats(data, dict_usecase, feature_set):
    # Add statistical features
    # Adding statistical features, default are: MEAN, STD, MAX, MIN, 'SKEWNESS', 'MOMENT', 'KURTOSIS' over the
    #    defined window size
    stat_feats = dict_usecase.get('stat_feats', [])
    print("Creating statistical features")
    data = create_statistical_features(data, features_to_select=stat_feats,
                                                   window_size=dict_usecase.get('stat_ws', 24))
    statistical_feature_names = [f"{name}_{stat_name}" for name in stat_feats for stat_name in
                                 dict_usecase.get('stat_vals', [])]
    feature_set.add_static_input_features(statistical_feature_names, feature_set.get_output_feature_names())
    return data, feature_set