from sklearn.base import BaseEstimator, TransformerMixin
from . import FitTransform, PickleInterface, FeatureNames


class BasicTransformer(BaseEstimator, FitTransform, PickleInterface, FeatureNames, TransformerMixin):
    pass