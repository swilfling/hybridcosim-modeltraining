import pandas as pd
from ..feature_engineering.featureengineeringbasic.featurecreators import CategoricalFeatures, CyclicFeatures
from ..feature_engineering.featureset import FeatureSet
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
    for name in cyclic_feat_cr.get_additional_feat_names() + categorical_feat_cr.get_additional_feat_names():
        feature_set.add_cyclic_input_feature(name)
    return feature_set


def add_features_to_data(data: pd.DataFrame, dict_usecase: dict):
    """
    Add cyclic, categorically encoded and statistical features to the feature set.
    @param data: data set
    @param dict_usecase: Dict containing configuration
    @return: extended data set
    """

    cyclic_feat_cr = CyclicFeatures(selected_feats=dict_usecase.get('cyclical_feats',[]))
    categoric_feat_cr = CategoricalFeatures(selected_feats=dict_usecase.get('onehot_feats', []))
    feature_creators = make_pipeline(cyclic_feat_cr, categoric_feat_cr, 'passthrough')
    return feature_creators.fit_transform(data)
