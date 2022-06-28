from .feature_expansion import FeatureExpansion
from sklearn.preprocessing import PolynomialFeatures


class PolynomialExpansion(FeatureExpansion):
    """
    Polynomial Feature Expansion
    Expands features by polynomials of variable order -
    https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    Implements scikit-learn's TransformerMixin interface.
    """
    def __init__(self, degree=2, interaction_only=False, include_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.model = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=include_bias)

    def _fit(self, X=None, y=None, **fit_params):
        self.model.fit(X, y)

    def _transform(self, x=None):
        return self.model.transform(x)

    def _get_feature_names_out(self, feature_names=None):
        return self.model.get_feature_names_out(feature_names)

