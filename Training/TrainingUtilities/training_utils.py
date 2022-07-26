import os

import numpy as np
import pandas as pd
import copy

from sklearn.model_selection import train_test_split

from ...Utilities import feature_combination as fc
from ...feature_engineering.parameters import TrainingParams, TrainingParamsExpanded
from ...feature_engineering.expandedmodel import ExpandedModel
from ModelTraining.feature_engineering.featureengineering.featureexpanders import PolynomialExpansion
from ...datamodels.datamodels.processing.shape import split_into_target_segments
from ...datamodels.datamodels import Model
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
    if isinstance(model, Model):
        model.save(model_dir)
    if isinstance(model, ExpandedModel):
        model.save_pkl(model_dir, "expanded_model.pkl")
    training_params.to_file(os.path.join(model_dir, f"parameters_{training_params.model_name}.json"))


def set_train_params_model(training_params_basic_config, feature_set, target_feature, model_type, transformer_params=None):
    """
    Set values of training params - Specific for use case with one target feature!
    """
    training_params = copy.copy(training_params_basic_config)
    training_params.model_type = model_type
    training_params.model_name = target_feature
    training_params.target_features = [target_feature]
    training_params.static_input_features = feature_set.get_static_input_feature_names(target_feature)
    training_params.dynamic_input_features = feature_set.get_dynamic_input_feature_names(target_feature)
    training_params.dynamic_output_features = feature_set.get_dynamic_output_feature_names(target_feature)
    if isinstance(training_params_basic_config, TrainingParamsExpanded) and transformer_params is not None:
        training_params.transformer_params = transformer_params
    return training_params


################################ Training and test set #######################################################


def expand_features(data: pd.DataFrame, feature_names, expander_parameters={}):
    """
    Expand features through polynomial expansion
    @param data: input data
    @param feature_names: names of features to expand
    @param expander_parameters: Parameters for polynomial expansion
    @return: dataframe containing expanded data
    """
    expander = PolynomialExpansion(**expander_parameters)
    data_expanded = expander.fit_transform(data[feature_names])
    feature_names_expanded = expander.get_feature_names_out(feature_names)
    return pd.DataFrame(data_expanded, columns=feature_names_expanded)


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

def extract_training_and_test_set(data: pd.DataFrame, training_params: TrainingParams):
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
    dynamic_features, y = split_into_target_segments(
        features=dynamic_features,
        targets=targets,
        lookback_horizon=lookback_horizon,
        prediction_horizon=prediction_horizon
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

