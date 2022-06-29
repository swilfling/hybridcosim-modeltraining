import numpy as np
from sklearn.model_selection import GridSearchCV
from ..feature_engineering.expandedmodel import ExpandedModel
from ..datamodels.datamodels import Model


def best_estimator(model: Model, x_train: np.ndarray, y_train: np.ndarray, parameters={}, cv_folds=5):
    """
    Grid search for best estimator
    @param model: object of class Model or ExpandedModel
    @param x_train: Training input (n_samples, lookback + 1, n_features)
    @param y_train: Ground truth (n_samples, n_target_features)
    @parameters: Grid search parameters
    @return: Best parameters - dict
    """
    # Transform x train
    search = GridSearchCV(model.get_estimator(), parameters, cv=cv_folds, scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],refit='r2', verbose=4)
    x_train = model.reshape_data(x_train)
    x_train, y_train = model.scale(x_train, y_train)
    if isinstance(model, ExpandedModel):
        x_train = model.transformers.fit_transform(x_train, y_train)
    search.fit(x_train, y_train)
    if isinstance(model, ExpandedModel):
        print(f"Best score for model {model.__class__.__name__} - {model.model.__class__.__name__} is: {search.best_score_}")
    else:
        print(f"Best score for model {model.__class__.__name__} is: {search.best_score_}")
    print(f"Best parameters are {search.best_params_}")
    return search.best_params_


def best_pipeline(model: ExpandedModel, x_train: np.ndarray, y_train: np.ndarray, parameters={}, cv_folds=5):
    """
    Grid search for best pipeline
    @param model: object of class Model or ExpandedModel
    @param x_train: Training input (n_samples, lookback + 1, n_features)
    @param y_train: Ground truth (n_samples, n_target_features)
    @parameters: Grid search parameters
    @return: Best parameters - dict
    """
    search = GridSearchCV(model.get_full_pipeline(), parameters, cv=cv_folds, scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],refit='r2', verbose=4)
    # Transform x train
    x_train, y_train = model.preprocess(x_train, y_train)
    search.fit(x_train, y_train)
    print(f"Best score for model {model.__class__.__name__} - {model.model.__class__.__name__} is: {search.best_score_}")
    print(f"Best parameters are {search.best_params_}")
    return search.best_params_
