from sklearn.base import BaseEstimator
from ..interfaces import FitTransform, PickleInterface, FeatureNames


class BasicTransformer(BaseEstimator, FitTransform, PickleInterface, FeatureNames):
    pass