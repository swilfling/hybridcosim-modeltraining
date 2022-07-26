from ..featureengineering.interfaces import BasicTransformer, BasicInterface


class TransformerWrapper(BasicTransformer):
    """
    Transform only selected features - keep other features
    Transformed features replace basic features
    """
    transformer_type = ""
    transformer_params = {}
    transformer_ = None

    def __init__(self, transformer_type="", transformer_params={}, **kwargs):
        self.transformer_type = transformer_type
        self.transformer_params = transformer_params

    def _fit(self, X, y, **fit_params):
        self.transformer_ = BasicInterface.from_name(self.transformer_type, **self.transformer_params)
        if self.transformer_ is not None:
            self.transformer_.fit(X, y, **fit_params)

    def _transform(self, X):
        return self.transformer_.transform(X)

    def _get_feature_names_out(self, feature_names=None):
        return self.transformer_.get_feature_names_out(feature_names) if self.transformer_ is not None else None