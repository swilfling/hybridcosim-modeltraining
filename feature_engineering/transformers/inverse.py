from ..featureengineeringbasic.interfaces import BasicTransformer


class InverseTransform(BasicTransformer):
    """
    Inverse transformation
    """
    def transform(self, X):
        X_tr = X.copy()
        X_tr[X_tr != 0] = 1 / (X_tr[X_tr != 0])
        return X_tr

    def _get_feature_names_out(self, feature_names=None):
        return [f"{name}_inv" for name in feature_names] if feature_names is not None else None