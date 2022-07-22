from . import FeatureExpansion
from sklearn.preprocessing import PolynomialFeatures


class PolynomialExpansion(FeatureExpansion):
    """
    Polynomial Feature Expansion
    Expands features by polynomials of variable order -
    https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    Implements scikit-learn's TransformerMixin interface.
    """
    degree = 2
    interaction_only=False
    include_bias=True
    model_ = None

    def __init__(self, degree=2, interaction_only=False, include_bias=True, **kwargs):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def _fit(self, X=None, y=None, **fit_params):
        self.model_ = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only, include_bias=self.include_bias)
        self.model_.fit(X, y)

    def _transform(self, x=None):
        return self.model_.transform(x)

    def _get_feature_names_out(self, feature_names=None):
        return self.model_.get_feature_names_out(feature_names)

