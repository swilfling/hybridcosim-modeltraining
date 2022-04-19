import logging

from os import makedirs
from tensorflow import keras
from typing import List

import ModelTraining.Training.predict
import ModelTraining.Utilities.Plotting.plot_training_results as plt_utils
from ModelTraining.Utilities.feature_set import FeatureSet

from datamodels import (
    RandomForestRegression,
    XGBoost,
    LinearRegression
)
from datamodels.processing import (
    IdentityScaler,
    Normalizer,
    RobustStandardizer,
    Standardizer,
)

import datamodels.processing.shape
from ModelTraining.TrainingUtilities.training_utils import *
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults


def setup_training():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)

    model_to_class = {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegression,
        "XGBoost": XGBoost,
    }

    normalizers_to_class = {
        None: IdentityScaler,
        "standardize": Standardizer,
        "normalize": Normalizer,
        "standardize-robust": RobustStandardizer,
    }

    metrics = None

    dict_training_setup = {"Normalizers": normalizers_to_class,
                           "ModelTypes": model_to_class,
                           "Logger": logger,
                           "Metrics": metrics}
    return dict_training_setup


def run_experiment(
        data,
        model: str,
        feature_set: FeatureSet,
        target_feature_names: List[str],
        lookback_horizon: int,
        prediction_horizon: int,
        epochs: int,
        training_split: float,
        validation_split: float,
        normalization: str,
        model_to_class: dict,
        normalizers_to_class: dict
):
     if model not in model_to_class:
        raise RuntimeError(
            f"The specified model name: '{model}' is not recognized. "
            f"Please select a one from the following: {model_to_class.keys()}"
        )

     target_feature_name = target_feature_names[0]
     static_feature_names = feature_set.get_static_feature_names(target_feature_name)
     dynamic_feature_names = feature_set.get_dynamic_feature_names(target_feature_name)

     training_feature_names = static_feature_names + dynamic_feature_names

     # Select features from data
     input_train, input_test = datamodels.processing.shape.split(
         data=datamodels.processing.numeric.interpolate(data[training_feature_names]),
         frac=training_split
     )
     target_train, target_test = datamodels.processing.shape.split(
         data=datamodels.processing.numeric.interpolate(data[target_feature_names]),
         frac=training_split,
     )

     # Static features - cut off first <lookback horizon> values
     input_train_static = input_train[static_feature_names].to_numpy()[lookback_horizon:]
     target_train_static = target_train.to_numpy()[lookback_horizon:]

     input_test_static = input_test[static_feature_names].to_numpy()[lookback_horizon:]
     target_test_static = target_test.to_numpy()[lookback_horizon:]

     # Static features - no lookback horizon, only values from current time
     x_train_static, _ = datamodels.processing.shape.split_into_target_segments(
         features=input_train_static,
         targets=target_train_static,
         lookback_horizon=0,
         prediction_horizon=prediction_horizon,
     )

     # Dynamic features - use lookback horizon
     x_train_dynamic, y_train = datamodels.processing.shape.split_into_target_segments(
         features=input_train[dynamic_feature_names].to_numpy(),
         targets=target_train.to_numpy(),
         lookback_horizon=lookback_horizon,
         prediction_horizon=prediction_horizon,
     )

     # For random forest, LR, NN models: Reshape training data to pass dynamic features as single vector
     x_train_dynamic = x_train_dynamic.reshape((
         x_train_dynamic.shape[0], 1, x_train_dynamic.shape[1] * x_train_dynamic.shape[2]))
     x_train = np.dstack([x_train_static, x_train_dynamic])

     print('xtrain', x_train.shape)
     print('ytrain', y_train.shape)

     x_test_static, _ = datamodels.processing.shape.split_into_target_segments(
         features=input_test_static,
         targets=target_test_static,
         lookback_horizon=0,
         prediction_horizon=prediction_horizon,
     )

     x_test_dynamic, y_test = datamodels.processing.shape.split_into_target_segments(
         features=input_test[dynamic_feature_names].to_numpy(),
         targets=target_test.to_numpy(),
         lookback_horizon=lookback_horizon,
         prediction_horizon=prediction_horizon,
     )

     # For random forest, LR, NN models: Reshape training data to pass dynamic features as single vector
     x_test_dynamic = x_test_dynamic.reshape((
         x_test_dynamic.shape[0], 1, x_test_dynamic.shape[1] * x_test_dynamic.shape[2]))
     x_test = np.dstack([x_test_static, x_test_dynamic])

     x_train_index = input_train.iloc[lookback_horizon + prediction_horizon:].index.to_series()
     x_test_index = input_test.iloc[lookback_horizon + prediction_horizon:].index.to_series()

     if x_train_index.shape[0] != y_train.shape[0]:
         ValueError(f"inferred index and target labels should be of same length.\n"
                    f"index: {x_train_index.shape[0]}, targets: {y_train.shape[0]}")
     if x_test_index.shape[0] != y_test.shape[0]:
         ValueError(f"inferred index and target labels should be of same length.\n"
                    f"index: {x_train_index.shape[0]}, targets: {y_train.shape[0]}")

     def network_train_function(model, x_train, y_train):
         return model.fit(
             x_train,
             y_train,
             epochs=epochs,
             validation_split=validation_split,
             callbacks=[
                 keras.callbacks.EarlyStopping(
                     monitor="val_loss", patience=30, restore_best_weights=True
                 )],
         )

     model_instance = model_to_class[model](
         x_scaler_class=normalizers_to_class[normalization],
         train_function=network_train_function,
     )

     logging.info(
         f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}"
     )

     model_instance.train(x_train, y_train)
     y_pred_train = ModelTraining.Training.predict.predict(x_train)
     y_pred_test = ModelTraining.Training.predict.predict(x_test)

     return TrainingParams(
         model_name=model,
         static_input_features=static_feature_names,
         dynamic_input_features=dynamic_feature_names,
         target_features=target_feature_names,
         lookback_horizon=lookback_horizon,
         prediction_horizon=prediction_horizon,

     ), TrainingResults(
         train_index=x_train_index,
         train_target=y_train,
         train_prediction=y_pred_train,
         test_index=x_test_index,
         test_target=y_test,
         test_prediction=y_pred_test
     ), model_instance


def run_training(data,
                 results_dir,
                 lookback_horizons: List[int],
                 prediction_horizons: List[int],
                 models: List[str],
                 feature_set: FeatureSet,
                 target_feature_names: List[str],
                 dict_training_setup: dict,
                 epochs=100,
                 normalization=None,
                 training_split=0.8,
                 validation_split=0.2):
    # Get training setup data
    model_to_class = dict_training_setup["ModelTypes"]
    normalizers_to_class = dict_training_setup["Normalizers"]
    logger = dict_training_setup["Logger"]

    # Create result dir
    if not os.path.exists(results_dir):
        makedirs(results_dir)

    # Define experiments
    experiments = [
        (m, l, p)
        for m in models
        for l in lookback_horizons
        for p in prediction_horizons
    ]

    logger.info(
        f"Commencing execution of '{len(experiments)}' experiment(s) using different combinations "
        f"models and features for training, results will be dumped in the '{results_dir}' directory."
    )
    for model, lookback_horizon, prediction_horzion in experiments:
        logger.info(
            f"Running experiment for the following model: '{model}'  with lookback {lookback_horizon} "
            f"and prediction horizon {prediction_horzion}."
        )
        try:
            for target_feature_name in target_feature_names:
                # Features for current model
                params, results, trained_model = run_experiment(
                    data=data,
                    model=model,
                    feature_set=feature_set,
                    target_feature_names=[target_feature_name],
                    lookback_horizon=lookback_horizon,
                    prediction_horizon=prediction_horzion,
                    epochs=epochs,
                    normalization=normalization,
                    training_split=training_split,
                    validation_split=validation_split,
                    model_to_class=model_to_class,
                    normalizers_to_class=normalizers_to_class
                )

                logger.info(f"Experiment executed successfully, saving results.")

                if not os.path.exists(results_dir / target_feature_name):
                    os.mkdir(results_dir / target_feature_name)

                params_filename = "parameters_" + target_feature_name
                params.to_file(results_dir / target_feature_name / params_filename)
                trained_model.save(results_dir / target_feature_name)

                figtitle = (f'Model: {model}, '
                            f'pred: {prediction_horzion}, '
                            f'lookback: {lookback_horizon}, '
                            f'feature: {target_feature_name}')

                output_filename = results_dir / f"{model}_{target_feature_name}.png"
                plt_utils.plot_training_results(results, figtitle, output_filename)

                logger.info("results updated, proceeding to next experiment.")

        except Exception:
            logger.error(f"An runtime error has ocurred during the experiment using model: '{model}' "
                         f"Skipping to next model, no results will be stored for this model. "
                         f"Please check the stack trace and fix the error.",
                         exc_info=True)
    logger.info(
        f"Finished all experiments, results for the individual "
        f"experiments can be found in the '{results_dir}' directory.")
