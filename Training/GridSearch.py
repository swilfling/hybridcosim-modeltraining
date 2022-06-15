from sklearn.model_selection import GridSearchCV
from ..datamodels.datamodels.wrappers.feature_extension import ExpandedModel


def best_estimator(model, x_train, y_train, parameters={}):
    estimator = model.get_estimator()
    search = GridSearchCV(estimator, parameters, scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],refit='r2', verbose=4)
    # Transform x train
    if x_train.ndim == 3:
        x_train = x_train.reshape(x_train.shape[0], -1)
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
    x_train = model.reshape_data(x_train)
    x_train, y_train = model.scale(x_train, y_train)
    if type(model) == ExpandedModel:
        x_train = model.transformers.fit_transform(x_train, y_train)
    search.fit(x_train, y_train)
    if type(model) == ExpandedModel:
        print(f"Best score for model {model.__class__.__name__} - {model.model.__class__.__name__} is: {search.best_score_}")
    else:
        print(f"Best score for model {model.__class__.__name__} is: {search.best_score_}")
    print(f"Best parameters are {search.best_params_}")
    return search.best_params_

