import logging
from typing import List
from .TrainingUtilities import training_utils as train_utils
from ..Training.TrainingUtilities.parameters import TrainingParams
from ..datamodels.datamodels.processing import datascaler
from ..datamodels import datamodels


def run_training_and_test(data, list_training_parameters: List[TrainingParams], prediction_type="History", **kwargs):
    """
    Function: run training and test - Train models based on training data - Train multiple models based on list of training params.
    Includes:
        Train-test split
        Feature expansion
        Feature selection
        Grid search
        Prediction

    Parameters:
        @param data: data to run training on
        @param list_training_parameters: list of training parameters
        @param prediction_type: Type of prediction - choose 'History' or 'ground truth'
        @param model_parameters: parameters for grid search
        @param expander_parameters: parameters for polynomial expansion

    Returns:
        @return models (trained models), results of prediction for each model
    """
    models, results = [], []
    for training_params in list_training_parameters:
        model, train_data = run_training(data, training_params)
        train_data.test_prediction = model.predict(train_data.test_input)
        models.append(model)
        results.append(train_data)
    return models, results


def run_training(data, training_params: TrainingParams):
    """
    Run training
    :param data: training data
    :param training_params: training parameters
    :return: model, training data
    """
    index, x, y, feature_names = train_utils.extract_training_and_test_set(data, training_params)
    train_data = train_utils.create_train_data(index, x, y, training_split=training_params.training_split)
    # Create model
    logging.info(f"Training model with input of shape: {train_data.train_input.shape} and targets of shape {train_data.train_target.shape}")
    model = getattr(datamodels, training_params.model_type)(
        x_scaler_class=getattr(datascaler, training_params.normalizer),
        name=training_params.str_target_feats(),
        parameters={})
    model.train(train_data.train_input, train_data.train_target)
    return model, train_data

