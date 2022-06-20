import pandas as pd
from ModelTraining.Preprocessing.FeatureCreation.categoricalfeatures import CategoricalFeatures
from ModelTraining.Preprocessing.FeatureCreation.cyclicfeatures import CyclicFeatures
from ModelTraining.Preprocessing.FeatureCreation.statistical_features import StatisticalFeatures
from ModelTraining.Preprocessing.featureset import Feature, FeatureSet


def add_features_to_featureset(feature_set: FeatureSet, dict_usecase: dict):
    """
    Add cyclic, categorically encoded and statistical features to the feature set.
    @param feature_set: feature set
    @param dict_usecase: Dict containing configuration
    @return: extended feature set
    """
    # Add one hot encoded features and cyclic features
    cyclic_feats = dict_usecase.get('cyclical_feats', [])
    onehot_feats = dict_usecase.get('onehot_feats',[])
    cyclic_feature_names = CyclicFeatures(cyclic_feats).get_additional_feat_names()
    onehot_feat_names = CategoricalFeatures(onehot_feats).get_additional_feat_names()
    for name in cyclic_feature_names + onehot_feat_names:
        feature_set.add_feature(Feature(name=name, static=True, cyclic=True, input=True, init=0, models=feature_set.get_output_feature_names()))
    # Add statistical features
    statistical_feature_names = StatisticalFeatures(dict_usecase.get('stat_feats',[]),
                                                    dict_usecase.get('stat_vals',[]),
                                                    dict_usecase.get('stat_ws', 1)).get_additional_feat_names()
    for name in statistical_feature_names:
        feature_set.add_feature(Feature(name=name, static=True, statistical=True, input=True, init=0, models=feature_set.get_output_feature_names()))
    return feature_set


def add_features_to_data(data: pd.DataFrame, dict_usecase: dict):
    """
    Add cyclic, categorically encoded and statistical features to the feature set.
    @param data: data set
    @param dict_usecase: Dict containing configuration
    @return: extended data set
    """
    data = CyclicFeatures().transform(data)
    data = CategoricalFeatures().transform(data)
    stat_feats = StatisticalFeatures(dict_usecase.get('stat_feats', []), dict_usecase.get('stat_vals', []),
                                     dict_usecase.get('stat_ws', 1))
    data = stat_feats.transform(data)
    return data
