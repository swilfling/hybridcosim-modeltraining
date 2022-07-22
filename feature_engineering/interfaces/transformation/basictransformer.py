from sklearn.base import BaseEstimator, TransformerMixin
from ModelTraining.feature_engineering.interfaces import BaseFitTransform, PickleInterface, FeatureNames


class BasicTransformer(BaseEstimator, BaseFitTransform, PickleInterface, FeatureNames, TransformerMixin):
    pass
