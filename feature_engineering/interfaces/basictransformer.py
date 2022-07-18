from sklearn.base import BaseEstimator, TransformerMixin
from . import BaseFitTransform, PickleInterface, FeatureNames


class BasicTransformer(BaseEstimator, BaseFitTransform, PickleInterface, FeatureNames, TransformerMixin):
    pass
