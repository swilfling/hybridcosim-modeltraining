import numpy as np
from sklearn.model_selection import GridSearchCV
from ..datamodels.datamodels.wrappers.expandedmodel import ExpandedModel
from ..datamodels.datamodels import Model
import pandas as pd


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
    estimator = model.model if isinstance(model, Model) else model.get_estimator()
    search = GridSearchCV(estimator, parameters, cv=cv_folds, scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],refit='r2', verbose=4)
    if isinstance(model, ExpandedModel):
        x_train, y_train = model.scale(x_train, y_train)
        model.fit_transformers(x_train, y_train)
        x_train = model.transform_features(x_train)
        x_train = x_train.values if isinstance(x_train, pd.DataFrame) else x_train
    else:
        if model.x_scaler is not None:
            x_train = model.x_scaler.fit(x_train).transform(x_train)
        if model.y_scaler is not None:
            y_train = model.y_scaler.fit(y_train).transform(y_train)
        #if hasattr(model, 'reshape_x'):
        #    x_train = model.reshape_x(x_train)
        #if hasattr(model, 'reshape_y'):
        #    y_train = model.reshape_y(y_train)
        #if hasattr(model, 'reshape'):
        #    x_train = model.reshape(x_train)
        #    y_train = model.reshape(y_train)
    if x_train.ndim > 2:
        x_train = x_train[:,0,:]
    if y_train.ndim > 2:
        y_train = y_train[:,0,:]
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
    x_train, y_train = model.scale(x_train, y_train)
    search.fit(x_train, y_train)
    print(f"Best score for model {model.__class__.__name__} - {model.model.__class__.__name__} is: {search.best_score_}")
    print(f"Best parameters are {search.best_params_}")
    return search.best_params_
