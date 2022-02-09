import logging
from typing import List
import pandas as pd
import numpy as np
import os

from numpy import asarray
import Utilities.Plotting.plotting_utilities as plt_utils
import ModelTrainingUtilities.training_utils as train_utils
from Utilities.Parameters import TrainingParams, TrainingResults
from sklearn.metrics import mean_absolute_error
import datamodels.datamodels
import datamodels.datamodels.processing
import keras

def train_model(model, x_train, y_train):
    optimizer = keras.optimizers.Adam(lr=0.01)
    model.compile(loss='mse',
              optimizer=optimizer)
    return model.fit(
        x_train, y_train,
        epochs=40,
        batch_size=24,
        validation_split=0.2,

    )


def create_model(training_params: TrainingParams, **kwargs):
    model_type = getattr(datamodels.datamodels, training_params.model_type)
    normalizer = getattr(datamodels.datamodels.processing, training_params.normalizer)
    expanders = [getattr(datamodels.datamodels.processing, expander_name) for expander_name in
                 training_params.expansion]
    return model_type(x_scaler_class=normalizer, name=str(training_params.target_features), train_function=train_model,
                       expander_classes=expanders, **kwargs)



def add_names_to_features(static_feature_names, static_row):
    return {name: val for name, val in zip(static_feature_names, list(static_row.flatten()))}


""" Prediction function using ground truth values for testing ML model"""


def predict(model, x, y_true, training_params):
    print(x.shape)
    y_pred = model.predict(x)
    labels = [f"predicted_{feature}" for feature in training_params.target_features]
    new_df = pd.DataFrame(y_true, columns=training_params.target_features)
    for index, label in enumerate(labels):
        new_df[label] = y_pred[:,index]
    return new_df


def run_training_and_test(data, list_training_parameters: List[TrainingParams],
                          results_dir_path, plot_dir_name="plots", do_predict=True,
                          **kwargs):
    models = []
    results = []

    for training_params in list_training_parameters:
        # Get data and reshape
        index, x, y = train_utils.extract_training_and_test_set(data, training_params)
        index_train, x_train, y_train, index_test, x_test, y_test = train_utils.split_into_training_and_test_set(
            index, x, y, training_params.training_split)
        # Train model
        logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")
        model = create_model(training_params, **kwargs)
        model.train(x_train, y_train)
        models.append(model)

        result = TrainingResults(train_index=index_train, train_target=y_train,
                                 test_index=index_test, test_target=y_test)
        if do_predict:
            result_prediction = predict(model, x_test, y_test, training_params)

            result.test_prediction = result_prediction[[f"predicted_{feature}" for feature in training_params.target_features]]
            title = f"{training_params.model_type}: {training_params.model_name}"
            plot_dir = os.path.join(results_dir_path, plot_dir_name)
            os.makedirs(plot_dir, exist_ok=True)
            plt_utils.plot_result(result_prediction, plot_dir, title)
            #plot_training_results(result, title, os.path.join(plot_dir, title))

        results.append(result)
        # Save Model
        train_utils.save_model_and_parameters(results_dir_path, model, training_params)

    return models, results