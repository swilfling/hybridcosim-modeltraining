import os
import random
import Utilities.type_casting as tc
import numpy as np
import pandas as pd
import copy
from Utilities.Parameters import TrainingParams
from ModelTraining.datamodels.datamodels.processing.shape import split_into_target_segments
from sklearn.model_selection import KFold


########################## Dataset processing #####################################


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
    #targets = np.expand_dims(data[training_params.target_features].to_numpy(), axis=-1)

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

    static_features = None
    if training_params.static_input_features:
        """
        STATIC FEATURES
        shape: number of samples, 1, number of features
        """
        static_features = data[training_params.static_input_features].to_numpy()
        static_features = static_features[lookback_horizon:static_features.shape[0] - prediction_horizon]
#        if training_params.do_polynom_expansion:
#            static_features, static_feature_names = tc.polynomial_expansion(static_features, training_params.static_input_features)
#            training_params.static_feature_names_expanded = static_feature_names
#        if training_params.do_spline_interpolation:
#            static_features = tc.spline_interpolation(x_train=static_features, y_train=targets)
        static_features = np.expand_dims(static_features, axis=1)

        """
        RESHAPED DYNAMIC FEATURES (to fit STATIC FEATURE shape)
        shape: number of samples, 1, (lookback horizon + 1) * number of features)
        """
        dynamic_features = dynamic_features.reshape(
            (dynamic_features.shape[0], 1, dynamic_features.shape[1] * dynamic_features.shape[2]))

    x = tc.combine_static_and_dynamic_features(static_features, dynamic_features)

    return  index, x, y


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


############################# Model saving and plotting ##############################

def save_model_and_parameters(results_main_dir, model, training_params: TrainingParams):
    model_dir = os.path.join(results_main_dir, training_params.model_name)
    # Save model and parameters
    model.save(model_dir)
    training_params.to_file(os.path.join(model_dir, f"parameters_{training_params.model_name}.json"))



def set_train_params_model(training_params_basic_config, feature_set, target_feature, model_type):
    training_params = copy.copy(training_params_basic_config)
    training_params.model_type = model_type
    training_params.model_name = target_feature
    training_params.target_features = [target_feature]
    training_params.static_input_features = feature_set.get_static_feature_names(target_feature)
    training_params.dynamic_input_features = feature_set.get_dynamic_feature_names(target_feature)
    return training_params