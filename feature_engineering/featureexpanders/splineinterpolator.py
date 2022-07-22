from . import FeatureExpansion
from sklearn.preprocessing import SplineTransformer


class SplineInterpolator(FeatureExpansion):
    """
    Spline Interpolation
    Expands features by spline bases -
     see https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.SplineTransformer.html
    Implements scikit-learn's TransformerMixin interface.
    """
    n_knots = 5
    degree = 3
    model_ = None

    def __init__(self, n_knots=5, degree=3, **kwargs):
        self.n_knots = n_knots
        self.degree = degree

    def _get_feature_names_out(self, feature_names=None):
        return self.model_.get_feature_names_out(feature_names)

    def _fit(self, X=None, y=None, **fit_params):
        self.model_ = SplineTransformer(n_knots=self.n_knots, degree=self.degree)
        self.model_.fit(X, y)

    def _transform(self, x=None):
        return self.model_.transform(x)

