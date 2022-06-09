from functools import reduce
from operator import concat
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from typing import List
from ..datamodels.datamodels.processing.feature_extension import FeatureExpansion
from ..Preprocessing.FeatureSelection.FeatureSelector import FeatureSelector
from ..datamodels.datamodels.model import Model


def prepare_data_for_fit(model:Model, x_train, y_train):
    if x_train.ndim == 3:
        x_train = x_train.reshape(x_train.shape[0], -1)
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
    x_train = model.reshape_data(x_train)
    return model.scale(x_train, y_train)


def best_estimator(model:Model, x_train, y_train, parameters={}):
    estimator = model.model
    search = GridSearchCV(estimator, parameters, scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],refit='r2', verbose=4)
    # Transform x train
    x_train, y_train = prepare_data_for_fit(model, x_train, y_train)
    # Fit grid search params
    pipeline = make_pipeline(*model.expanders.get_list_expanders(), search)
    pipeline.fit(x_train, y_train)

    print(f"Best score for model {model.__class__.__name__} is: {search.best_score_}")
    print(f"Best parameters are {search.best_params_}")
    return search.best_params_


def create_pipeline(expanders: List[FeatureExpansion]=[], selectors: List[FeatureSelector]=[], model='passthrough'):
    list_selectors = reduce(concat, [[expander, selector] for expander, selector in zip(expanders, selectors)])
    return make_pipeline(*list_selectors, model)
