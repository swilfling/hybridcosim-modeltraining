from .feature_expansion import FeatureExpansion
from sklearn.preprocessing import PolynomialFeatures


class PolynomialExpansion(FeatureExpansion):
    """
    Polynomial Feature Expansion
    Expands features by polynomials of variable order -
    https://scikit-learn.org/dev/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
    Implements scikit-learn's TransformerMixin interface.
    """
    def __init__(self, **kwargs):
        self.model = PolynomialFeatures(**kwargs)

    def _fit(self, x=None, y=None):
        self.model.fit(x, y)

    def _transform(self, x=None):
        return self.model.transform(x)

    def _get_feature_names(self, feature_names=None):
        return self.model.get_feature_names_out(feature_names)

