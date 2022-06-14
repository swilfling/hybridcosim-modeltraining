import logging
from typing import List

from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.Training.predict import predict_with_history
from ModelTraining.Training.GridSearch import fit_feature_selectors, best_estimator
from ModelTraining.Preprocessing.FeatureSelection.feature_selectors import FeatureSelector
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.Preprocessing.FeatureSelection.feature_selection_params import FeatureSelectionParams
from ModelTraining.datamodels.datamodels.processing.datascaler import DataScaler
from ModelTraining.datamodels.datamodels import Model
from ModelTraining.datamodels.datamodels.wrappers.feature_extension import ExpandedModel, ExpanderSet


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
        @return models (trained models), results of prediction for each model, and feature selectors
    """
    models, results, list_selectors = [], [], []
    # Get optional arguments
    model_params = kwargs.get('model_parameters', {})
    expander_parameters = kwargs.get('expander_parameters',{})
    feature_select_params = kwargs.get('feature_select_params', [FeatureSelectionParams()])

    for training_params in list_training_parameters:
        model, result, selectors = run_training_model(data, training_params, expander_parameters, model_params, feature_select_params, prediction_type)
        list_selectors.append(selectors)
        models.append(model)
        results.append(result)
    return models, [results, list_selectors]


def run_training_model(data, training_params=TrainingParams(), expander_parameters={}, model_parameters={},
                       feature_select_params=[FeatureSelectionParams()], prediction_type='History'):
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
            @param expander_parameters: parameters for polynomial expansion

        Returns:
            @return model, prediction result: Training results, feature selectors
        """
    index, x, y, feature_names = train_utils.extract_training_and_test_set(data, training_params)
    index_train, x_train, y_train, index_test, x_test, y_test = train_utils.split_into_training_and_test_set(
        index, x, y, training_params.training_split)

    # Create model
    logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")
    model_basic = Model.from_name(training_params.model_type, x_scaler_class=DataScaler.cls_from_name(training_params.normalizer),
                            name=training_params.str_target_feats(), parameters={})
    # Create expanded model
    model = ExpandedModel(expanders=ExpanderSet.from_names(training_params.expansion, **expander_parameters), model=model_basic)
    # Create and fit selectors
    selectors = [FeatureSelector.from_params(params) for params in feature_select_params]
    fit_feature_selectors(model, selectors, x_train, y_train)
    model.set_feature_names(feature_names)
    # Select features + Grid Search
    best_params = best_estimator(model, x_train, y_train, parameters=model_parameters)
    model.get_estimator().set_params(**best_params)
    # Train final model
    model.train(x_train, y_train)
    # Predict test data
    y_pred = predict_with_history(model, index_test, x_test, y_test, training_params) \
        if prediction_type == 'History' else model.predict(x_test)
    result = TrainingResults(train_index=index_train, train_target=y_train,
                                   test_index=index_test, test_target=y_test,
                                   test_prediction=y_pred, test_input=x_test,
                                   target_feat_names=training_params.target_features)
    return model, result, selectors