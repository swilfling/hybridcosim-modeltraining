from sklearn.base import BaseEstimator, TransformerMixin
from . import BaseFitTransform, FeatureNames
from ..storage import PickleInterface


class BasicTransformer(BaseEstimator, BaseFitTransform, PickleInterface, FeatureNames, TransformerMixin):
    pass
