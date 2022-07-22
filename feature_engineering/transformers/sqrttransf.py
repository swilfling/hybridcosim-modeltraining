import numpy as np
from ..featureengineeringbasic.interfaces import BasicTransformer


class SqrtTransform(BasicTransformer):
    """
    Square root transformation
    """
    omit_neg_vals = True

    def __init__(self, omit_neg_vals=True, **kwargs):
        self.omit_neg_vals = omit_neg_vals

    def transform(self, X):
        if self.omit_neg_vals:
            X_tr = X.copy()
            X_tr[X >= 0] = np.sqrt(X[X >= 0])
            return X_tr
        else:
            return np.sqrt(X)

    def _get_feature_names_out(self, feature_names=None):
        return [f"{name}_sqrt" for name in feature_names] if feature_names is not None else None
