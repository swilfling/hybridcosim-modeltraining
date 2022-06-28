from .feature_expansion import FeatureExpansion
from sklearn.preprocessing import SplineTransformer


class SplineInterpolator(FeatureExpansion):
    """
    Spline Interpolation
    Expands features by spline bases -
     see https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.SplineTransformer.html
    Implements scikit-learn's TransformerMixin interface.
    """
    def __init__(self, n_knots=5, degree=3, **kwargs):
        super().__init__(**kwargs)
        self.model = SplineTransformer(n_knots=n_knots, degree=degree)

    def _get_feature_names_out(self, feature_names=None):
        return self.model.get_feature_names_out(feature_names)

    def _fit(self, X=None, y=None, **fit_params):
        self.model.fit(X, y)

    def _transform(self, x=None):
        return self.model.transform(x)

