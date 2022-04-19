from typing import List

from numpy.core.multiarray import asarray
from sklearn.metrics import mean_absolute_error

import ModelTraining.TrainingUtilities.preprocessing
from ModelTraining.Training.predict import predict
from ModelTraining.Training.ModelCreation import create_model
from ModelTraining.Utilities.Parameters import TrainingParams


def run_training_ensemble(data, list_training_parameters: List[TrainingParams],
                          n_members):
    models = []
    results = []
    for training_params in list_training_parameters:
        # Get data and reshape
        index, x, y,_ = ModelTraining.ModelTrainingUtilities.preprocessing.extract_training_and_test_set(data, training_params)
        index_train, x_train, y_train, index_test, x_test, y_test = ModelTraining.ModelTrainingUtilities.preprocessing.split_into_training_and_test_set(index, x, y, training_params.training_split)
        ensemble = list()
        for i in range(n_members):
            # define and fit the model on the training set
            model = create_model(training_params)
            model.train(x_train, y_train)
            # evaluate model on the test set
            yhat = predict(model, index_test, x_test, y_test)["predicted"]
            print("y_test", y_test)
            print("y_hat", yhat)
            mae = mean_absolute_error(y_test, yhat)
            print('>%d, MAE: %.3f' % (i + 1, mae))
            # store the model
            ensemble.append(model)

        lower, upper, mean = [], [], []
        for index, row in enumerate(x_test[0:1000]):
            x_point = asarray([x_test[index, :]])
            yhat_new = [predict(model, index_test, x_point, y_test)["predicted"] for model in ensemble]
            yhat_new = asarray(yhat_new)

            # calculate 95% gaussian prediction interval
            interval = 2.58 * yhat_new.std()
            lower_point = yhat_new.mean() - interval
            lower.append(lower_point)
            upper_point = yhat_new.mean() + interval
            upper.append(upper_point)
            mean.append(yhat_new.mean())
            print("x_test lower bound", lower[index])
            print("x_test upper bound", upper[index])
            print("true value", y_test[index])
            print('95%% prediction interval: [%.3f, %.3f]' % (lower[index], upper[index]))
        return lower, mean, upper, y_test[0:1000]