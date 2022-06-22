from .feature_expansion import FeatureExpansion
from sklearn.preprocessing import SplineTransformer


class SplineInterpolator(FeatureExpansion):
    """
    Spline Interpolation
    Expands features by spline bases -
     see https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.SplineTransformer.html
    Implements scikit-learn's TransformerMixin interface.
    """
    def __init__(self, **kwargs):
        self.model = SplineTransformer(**kwargs)

    def _get_feature_names(self, feature_names=None):
        return self.model.get_feature_names_out(feature_names)

    def _fit(self, x=None, y=None):
        self.model = self.model.fit(x, y)

    def _transform(self, x=None):
        return self.model.transform(x)

