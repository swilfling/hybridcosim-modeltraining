from sklearn.model_selection import GridSearchCV


def prepare_data_for_fit(model, x_train, y_train):
    if x_train.ndim == 3:
        x_train = x_train.reshape(x_train.shape[0], -1)
    if y_train.ndim == 2 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
    x_train = model.reshape_data(x_train)
    x_train, y_train = model.scale(x_train, y_train)
    for expander in model.expanders:
        x_train = expander.fit_transform(x_train, y_train)
    return x_train, y_train


def best_estimator(model, x_train, y_train, parameters={}):
    estimator = model.model
    search = GridSearchCV(estimator, parameters, scoring=['r2','neg_mean_squared_error','neg_mean_absolute_error'],refit='r2', verbose=4)
    # Transform x train
    x_train, y_train = prepare_data_for_fit(model, x_train, y_train)
    # Fit grid search params
    search_result = search.fit(x_train, y_train)
    print(f"Best score for model {model.__class__.__name__} is: {search_result.best_score_}")
    print(f"Best parameters are {search_result.best_params_}")
    return search_result.best_params_