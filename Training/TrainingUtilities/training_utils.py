import json
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from ...Utilities import feature_combination as fc
from ...Training.TrainingUtilities.parameters import TrainingParams
from ...datamodels.datamodels.processing.shape import get_windows
from ...Utilities.trainingdata import TrainingData


############################# Model saving and plotting ##############################

def save_model_and_params(model, training_params: TrainingParams, results_main_dir: str="./"):
    """
    Save model and training parameters
    @param results_main_dir: output directory
    @param model: Model to save
    @param training_params: training parameters
    """
    model_dir = os.path.join(results_main_dir, training_params.model_name)
    os.makedirs(model_dir, exist_ok=True)
    # Save model and parameters
    model.save(model_dir)
    training_params.to_file(os.path.join(model_dir, f"parameters_{training_params.model_name}.json"))


################################ Training and test set #######################################################


def split_into_training_and_test_set(index, x, y, training_split=0.8, shuffle=False):
    """
    Split data into training and test set
    @param index: index (n_samples)
    @param x: input data (n_samples, lookback + 1, n_features)
    @param y: target values (n_samples, n_target_features)
    @training_split: fraction of data to use in training set
    @param shuffle: use random sampling
    @return: separated data
    """
    index_train, index_test, x_train, x_test, y_train, y_test = train_test_split(index, x, y, train_size=training_split, shuffle=shuffle)
    return index_train, x_train, y_train, index_test, x_test, y_test


def create_train_data(index, x, y, training_split=0.8, shuffle=False):
    """
    Split data into training and test set
    @param index: index (n_samples)
    @param x: input data (n_samples, lookback + 1, n_features)
    @param y: target values (n_samples, n_target_features)
    @training_split: fraction of data to use in training set
    @param shuffle: use random sampling
    @return: separated data
    """
    index_train, index_test, x_train, x_test, y_train, y_test = train_test_split(index, x, y, train_size=training_split,
                                                                                 shuffle=shuffle)
    return TrainingData(train_index=index_train, test_index=index_test, train_target=y_train, test_target=y_test,
                        train_input=x_train, test_input=x_test)


def extract_training_and_test_set(data: pd.DataFrame, training_params: TrainingParams, create_df=False):
    """
    Extract training and test set
    @param data: full dataset
    @param training_params: training parameters
    @return: index, x, y, feature names
    """
    lookback_horizon = training_params.lookback_horizon
    prediction_horizon = training_params.prediction_horizon

    index = data.iloc[lookback_horizon + prediction_horizon:].index.to_series()

    num_targets = len(training_params.target_features)
    num_samples = data.shape[0]
    targets = np.reshape(data[training_params.target_features].to_numpy(), (num_samples, num_targets))

    """
    DYNAMIC FEATURES
    shape: number of samples, lookback horizon + 1, number of features
    """
    dynamic_feat_names = training_params.dynamic_input_features + training_params.dynamic_output_features
    dynamic_features = data[dynamic_feat_names].to_numpy()
    dynamic_features, y = get_windows(
        features=dynamic_features,
        targets=targets,
        lookback=lookback_horizon,
        lookahead=prediction_horizon
    )
    dynamic_feat_names_lag = [f'{name}_{lag}' for lag in range(1,lookback_horizon+1) for name in dynamic_feat_names]
    dynamic_feature_names_full = dynamic_feat_names + dynamic_feat_names_lag

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

    x = fc.combine_static_and_dynamic_features(static_features, dynamic_features)
    feature_names = training_params.static_input_features + dynamic_feature_names_full

    if create_df:
        x = pd.DataFrame(index=index, data=x[:, 0, :], columns=feature_names)
        y = pd.DataFrame(index=index, data=y[:, 0, :], columns=training_params.target_features)
    return index, x, y, feature_names


########################## Load from JSON ########################################

def load_from_json(filename):
    with open(filename) as f:
        return json.load(f)