from . import TransformerSet
from ..interfaces import PickleInterface
from sklearn.pipeline import make_pipeline


class ExpandedModel(PickleInterface):
    """
    Expansion Model - contains set of feature expanders
    """
    model = None
    transformers: TransformerSet
    feature_names = None
    num_predictors = 0

    def __init__(self, model, transformers: TransformerSet, feature_names=None):
        self.model = model
        self.transformers = transformers
        self.feature_names = feature_names

    def train(self, X, y, **fit_params):
        """
        Predict - prediction with feature expansion
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @param y: tensor of shape (n_samples, input_features)
        """
        self.model.input_shape = X.shape[1:]
        self.num_predictors = X.shape[-1]
        X, y = self.preprocess(X, y)
        X = self.transformers.fit_transform(X, y)
        self.model.train_model(X, y)

    def reshape_data(self, X):
        return self.model.reshape_data(X)

    def scale(self, X, y):
        return self.model.scale(X, y)

    def preprocess(self, X, y=None):
        """
        Call preprocessing functions - used for training
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @param y: tensor of shape (n_samples, input_features)
        """
        #if not X.ndim == 3:
        #    raise RuntimeError(f'x must be an array of shape (samples, lookback_horizon + 1, input_features)\n'
        #                       f'but is {X.shape}')
        X = self.model.reshape_data(X)
        return self.model.scale(X, y)

    def fit(self, X, y, **fit_params):
        """
        Predict - prediction with feature expansion
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @param y: tensor of shape (n_samples, input_features)
        """
        self.train(X, y, **fit_params)

    def predict(self, X):
        """
        Predict - prediction with feature expansion
        @param X: np.array of shape (n_samples, lookback_horizon + 1, input_features)
        @return: np.array of shape (n_samples, input_features)
        """
        #if not X.ndim == 3:
        #    raise RuntimeError(f'x must be an array of shape (samples, lookback_horizon + 1, input_features)\n'
        #                       f'but is {X.shape}')
        X = self.model.reshape_data(X)
        X = self.model.x_scaler.transform(X)
        X = self.transformers.transform(X)
        y = self.model.predict_model(X)
        if not y.shape[0] == X.shape[0]:
            raise AssertionError(f'samples in prediction do not match samples in input\n'
                                 f'expected: {X.shape[0]}, but is {y.shape[0]}.')

        if not y.ndim == 2:
            raise AssertionError(f'predictions must be two dimensional (samples, target_features)\n'
                                 f'but are {y.shape}.')
        return self.model.y_scaler.inverse_transform(y)

    def fit_transformers(self, X, y, **fit_params):
        """
        Fit transformers.
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @param y: tensor of shape (n_samples, input_features)
        """
        X, y = self.preprocess(X, y)
        self.transformers.fit(X, y)

    def transform_features(self, X):
        """
        Transform features - requires fitted transformers.
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @return: transformed features
        """
        return self.transformers.transform(X)

    def get_transformed_feature_names(self):
        """
        Get expanded feature names - get feature names after all expansion steps
        """
        return self.transformers.get_feature_names_out(self.feature_names)

    def set_feature_names(self, feature_names=None):
        """
        Set feature names for model
        """
        self.feature_names = feature_names
        self.model.set_feature_names(self.get_transformed_feature_names())

    def get_estimator(self):
        return self.model.get_estimator()

    def get_num_predictors(self):
        return self.num_predictors

    @property
    def name(self):
        return self.model.name

    @name.setter
    def name(self, name=""):
        self.model.name = name

    def get_full_pipeline(self):
        """
        Create pipeline of transformers and estimators
        @return: pipeline
        """
        transformers = self.transformers.get_list_transfomers()
        estimator = self.model.get_estimator()
        return make_pipeline(*transformers, estimator)


