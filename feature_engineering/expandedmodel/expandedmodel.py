from . import TransformerSet
from ..interfaces import PickleInterface
from sklearn.pipeline import make_pipeline
import numpy as np


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
        self.num_predictors = X.shape[-1]
        X, y = self.scale(X, y)
        X = self.transformers.fit_transform(X, y)
        X = self.expand_x(X)
        y = self.expand_y(y)
        self.model.train_model(X, y)

    def scale(self, X, y):
        if self.model.x_scaler is not None:
            X = self.model.x_scaler.fit(X).transform(X)
        if self.model.y_scaler is not None:
            y = self.model.y_scaler.fit(y).transform(y)
        return X, y

    def preprocess(self, X, y=None, use_transformers=True):
        """
        Call all preprocessing functions
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @param y: tensor of shape (n_samples, input_features)
        @param use_transformers: Use transformers or not
        @return: X, y preprocessed
        """
        X, y = self.scale(X, y)
        if use_transformers:
            X = self.transformers.fit_transform(X, y)
        return self.reshape(X, y)

    def expand_x(self, X):
        if X.ndim != 3:
            return np.expand_dims(X, axis=1)
        return X

    def expand_y(self, y):
        if y.ndim == 1:
            return np.reshape(y,(y.shape[0], 1, 1))
        if y.ndim == 2:
            return np.expand_dims(y, axis=1)
        return y

    def reshape(self, X, y):
        X = self.expand_x(X)
        y = self.expand_y(y)
        if hasattr(self.model, 'reshape'):
            X = self.model.reshape(X)
            y = self.model.reshape(y)
        else:
            if hasattr(self.model, 'reshape_x'):
                X = self.model.reshape_x(X)
            if hasattr(self.model, 'reshape_y'):
                y = self.model.reshape_y(y)
        return X, y

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
        X = self.model.x_scaler.transform(X)
        X = self.transform_features(X)
        X = self.expand_x(X)
        y = self.model.predict_model(X)
        if not y.shape[0] == X.shape[0]:
            raise AssertionError(f'samples in prediction do not match samples in input\n'
                                 f'expected: {X.shape[0]}, but is {y.shape[0]}.')

        if y.ndim == 1:
            y = np.expand_dims(y, axis=-1)
        return self.model.y_scaler.inverse_transform(y)

    def fit_transformers(self, X, y, **fit_params):
        """
        Fit transformers.
        @param X: tensor of shape (n_samples, lookback_horizon + 1, input_features)
        @param y: tensor of shape (n_samples, input_features)
        """
        X, y = self.scale(X, y)
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
        return self.model.model

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
        estimator = self.get_estimator()
        return make_pipeline(*transformers, estimator)


