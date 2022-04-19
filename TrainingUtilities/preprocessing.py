import random

import numpy as np
import pandas as pd

from ModelTraining.datamodels.datamodels.processing.shape import split_into_target_segments
from ModelTraining.Utilities import type_casting as tc


def split_into_training_and_test_set_random(index, x, y, training_split):
    rnd_ind = random.sample(range(0, x.shape[0]), x.shape[0])
    num_training_samples = int(training_split * x.shape[0])
    train_rnd_ind = rnd_ind[0:num_training_samples]
    test_rnd_ind = rnd_ind[num_training_samples:]

    x_train = x[train_rnd_ind]
    y_train = y[train_rnd_ind]
    x_test = x[test_rnd_ind]
    y_test = y[test_rnd_ind]

    index_train = index[train_rnd_ind]
    index_test = index[test_rnd_ind]
    return index_train, x_train, y_train, index_test, x_test, y_test


def split_into_training_and_test_set(index, x, y, training_split):
    num_training_samples = int(training_split * x.shape[0])

    x_train = x[:num_training_samples]
    y_train = y[:num_training_samples]

    x_test = x[num_training_samples:]
    y_test = y[num_training_samples:]

    index_train = index[:num_training_samples]
    index_test = index[num_training_samples:]

    return index_train, x_train, y_train, index_test, x_test, y_test


def extract_training_and_test_set(data, training_params):
    lookback_horizon = training_params.lookback_horizon
    prediction_horizon = training_params.prediction_horizon

    index = data.iloc[lookback_horizon + prediction_horizon:].index.to_series()

    num_targets = len(training_params.target_features)
    num_samples = data.shape[0]
    targets = np.reshape(data[training_params.target_features].to_numpy(), (num_samples,num_targets))

    """
    DYNAMIC FEATURES
    shape: number of samples, lookback horizon + 1, number of features
    """
    dynamic_features = data[training_params.dynamic_input_features].to_numpy()
    dynamic_features, y = split_into_target_segments(
        features=dynamic_features,
        targets=targets,
        lookback_horizon=lookback_horizon,
        prediction_horizon=prediction_horizon
    )
    dynamic_feature_names = create_dynamic_feature_names(training_params.dynamic_input_features,
                                                                  lookback_horizon)

    static_features = None
    if training_params.static_input_features:
        """
        STATIC FEATURES
        shape: number of samples, 1, number of features
        """
        static_features = data[training_params.static_input_features].to_numpy()
        static_features = static_features[lookback_horizon:static_features.shape[0] - prediction_horizon]
        static_features = np.expand_dims(static_features, axis=1)

        """
        RESHAPED DYNAMIC FEATURES (to fit STATIC FEATURE shape)
        shape: number of samples, 1, (lookback horizon + 1) * number of features)
        """
        dynamic_features = dynamic_features.reshape(
            (dynamic_features.shape[0], 1, dynamic_features.shape[1] * dynamic_features.shape[2]))

    x = tc.combine_static_and_dynamic_features(static_features, dynamic_features)
    feature_names = training_params.static_input_features + dynamic_feature_names
    return index, x, y, feature_names

"""
Replace the ground truth by the predicted values for all the target features in the dataset and save the new one
Assuming that each csv file contains values of one column (each model predicts one target feature without replicas)
"""

def replace_dataset(data, list_training_parameters, first_train_results_path, lookback_horizon):
    new_dataset = data[lookback_horizon + 1:]

    """
    Predicted dataset starts from (lookback horizon + 1)
    """
    for training_parameters in list_training_parameters:

        for feature in training_parameters.target_features:
            val = pd.read_csv(f"{first_train_results_path}/{training_parameters.model_type}_{feature}_predictions.csv", sep=',', engine='python').set_index('date')
            new_dataset = new_dataset.drop(feature, axis=1)
            new_dataset[feature] = val['predicted']

    return new_dataset


def create_dynamic_feature_names(dynamic_feature_names, lookback_horizon):
    return dynamic_feature_names + [f'{name}_{lag}' for lag in range(1,lookback_horizon+1) for name in dynamic_feature_names]
