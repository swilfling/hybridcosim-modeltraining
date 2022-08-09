import logging
from typing import List
from .TrainingUtilities import training_utils as train_utils
from .predict import predict_with_history
from .GridSearch import best_estimator
from ..Training.TrainingUtilities.parameters import TrainingParams, TrainingParamsExpanded
from ..Utilities.trainingdata import TrainingData
from ..datamodels.datamodels.processing import datascaler
from ..datamodels import datamodels
from ..datamodels.datamodels.wrappers.expandedmodel import ExpandedModel, TransformerSet


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
    # Get optional arguments
    model_params = kwargs.get('model_parameters', {})

    for training_params in list_training_parameters:
        model, result, = run_training_model(data, training_params, model_params, prediction_type)
        models.append(model)
        results.append(result)
    return models, results


def run_training_model(data, training_params=TrainingParams(), model_parameters={}, prediction_type='History'):
    """
        Function: run training and test - Train models based on training data.
        Includes:
            Train-test split
            Feature expansion
            Feature selection
            Grid search
            Prediction

        Parameters:
            @param data: data to run training on
            @param training_params: list of training parameters
            @param prediction_type: Type of prediction - choose 'History' or 'ground truth'
            @param model_parameters: parameters for grid search

        Returns:
            @return model, prediction result: model, Training results
        """
    index, x, y, feature_names = train_utils.extract_training_and_test_set(data, training_params)
    index_train, x_train, y_train, index_test, x_test, y_test = train_utils.split_into_training_and_test_set(
        index, x, y, training_params.training_split)

    # Create model
    logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")
    model = getattr(datamodels, training_params.model_type)(
                                  x_scaler_class=getattr(datascaler, training_params.normalizer),
                                  name=training_params.str_target_feats(),
                                  parameters={})
    # Create expanded model wrapper
    if isinstance(training_params, TrainingParamsExpanded):
        model = ExpandedModel(transformers=TransformerSet.from_list_params(training_params.transformer_params),
                              model=model, feature_names=feature_names)
    # Select features + Grid Search
    best_params = best_estimator(model, x_train, y_train, parameters=model_parameters)
    model.get_estimator().set_params(**best_params)
    # Train final model
    model.train(x_train, y_train)
    # Predict test data
    y_pred = predict_with_history(model, index_test, x_test, y_test, training_params) \
        if prediction_type == 'History' else model.predict(x_test)
    result = TrainingData(train_index=index_train,
                          train_target=y_train,
                          test_index=index_test,
                          test_target=y_test,
                          test_prediction=y_pred,
                          test_input=x_test,
                          target_feat_names=training_params.target_features)
    return model, result