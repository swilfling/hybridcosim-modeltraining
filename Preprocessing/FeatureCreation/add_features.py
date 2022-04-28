from ModelTraining.Preprocessing.FeatureCreation.cyclic_features import add_cyclical_features
from ModelTraining.Preprocessing.FeatureCreation.feature_creation import create_additional_feats
from ModelTraining.Preprocessing.FeatureCreation.statistical_features import create_statistical_features
from ModelTraining.Preprocessing.feature_set import Feature

def add_features_to_data(data, usecase_name, stat_feat_names, stat_feat_types, stat_ws):
    data = create_additional_feats(data, usecase_name)
    data = add_cyclical_features(data)
    data = create_statistical_features(data, features_to_select=stat_feat_names, statistical_features=stat_feat_types,
                                       window_size=stat_ws)
    return data


def add_features_to_featureset(dict_usecase, feature_set):
    # Add one hot encoded features and cyclic features
    hour_1hot_feat = [f'hour_{h}' for h in range(24)]
    weekday_1hot_feat = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
    cyclic_feature_names = dict_usecase.get('cyclical_feats', [])
    if 'hours' in dict_usecase.get('onehot_feats', []):
        cyclic_feature_names += hour_1hot_feat
    if 'weekdays' in dict_usecase.get('onehot_feats', []):
        cyclic_feature_names += weekday_1hot_feat
    for name in cyclic_feature_names:
        feature_set.add_feature(Feature(name=name, static=True, cyclic=True, feature_type='input', init=0, models=feature_set.get_output_feature_names()))
    # Add statistical features
    statistical_feature_names = [f"{name}_{stat_name}" for name in dict_usecase.get('stat_feats',[]) for stat_name in
     dict_usecase.get('stat_vals', [])]
    for name in statistical_feature_names:
        feature_set.add_feature(Feature(name=name, static=True, cyclic=True, feature_type='input', init=0, models=feature_set.get_output_feature_names()))
    return feature_set


def add_features(data, feature_set, dict_usecase):
    data = add_features_to_data(data, dict_usecase['name'], dict_usecase['stat_feats'], dict_usecase['stat_vals'], dict_usecase['stat_ws'])
    feature_set = add_features_to_featureset(dict_usecase, feature_set)
    return data, feature_set