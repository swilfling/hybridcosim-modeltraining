import pandas as pd
from .featurecreators import CategoricalFeatures, CyclicFeatures, StatisticalFeatures
from ..featureset import FeatureSet
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
    statistical_feat_cr = StatisticalFeatures(selected_feats=dict_usecase.get('stat_feats', []), statistical_features=dict_usecase.get('stat_vals', []),
                                              window_size=dict_usecase.get('stat_ws', 1))
    for name in cyclic_feat_cr.get_additional_feat_names() + categorical_feat_cr.get_additional_feat_names():
        feature_set.add_cyclic_input_feature(name)
    # Add statistical features
    for name in statistical_feat_cr.get_additional_feat_names():
        feature_set.add_statistial_input_feature(name)
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
