import pandas as pd
from ModelTraining.Preprocessing.FeatureCreation.categoricalfeatures import CategoricalFeatures
from ModelTraining.Preprocessing.FeatureCreation.cyclicfeatures import CyclicFeatures
from ModelTraining.Preprocessing.FeatureCreation.statistical_features import StatisticalFeatures
from ModelTraining.Preprocessing.featureset import Feature, FeatureSet
from sklearn.pipeline import make_pipeline


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
    cyclic_feat_cr = CyclicFeatures(cyclic_feats)
    categorical_feat_cr = CategoricalFeatures(onehot_feats)
    statistical_feat_cr = StatisticalFeatures(dict_usecase.get('stat_feats', []),
                                              dict_usecase.get('stat_vals', []),
                                              dict_usecase.get('stat_ws', 1))
    for name in cyclic_feat_cr.get_additional_feat_names() + categorical_feat_cr.get_additional_feat_names():
        feature_set.add_feature(Feature(name=name, static=True, cyclic=True, input=True, init=0, models=feature_set.get_output_feature_names()))
    # Add statistical features
    for name in statistical_feat_cr.get_additional_feat_names():
        feature_set.add_feature(Feature(name=name, static=True, statistical=True, input=True, init=0, models=feature_set.get_output_feature_names()))
    return feature_set


def add_features_to_data(data: pd.DataFrame, dict_usecase: dict):
    """
    Add cyclic, categorically encoded and statistical features to the feature set.
    @param data: data set
    @param dict_usecase: Dict containing configuration
    @return: extended data set
    """
    cyclic_feat_cr = CyclicFeatures(dict_usecase.get('cyclical_feats', []))
    categorical_feat_cr = CategoricalFeatures(dict_usecase.get('onehot_feats', []))
    statistical_feat_cr = StatisticalFeatures(dict_usecase.get('stat_feats', []),
                                              dict_usecase.get('stat_vals', []),
                                              dict_usecase.get('stat_ws', 1))
    feature_creators = make_pipeline(cyclic_feat_cr, categorical_feat_cr, statistical_feat_cr, 'passthrough')
    return feature_creators.transform(data)
