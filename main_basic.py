import os
import logging

from ModelTraining.TrainingUtilities.MetricsExport import MetricsExport
import ModelTraining.Utilities.DataProcessing.data_preprocessing as dp_utils
import ModelTraining.Utilities.DataProcessing.data_import as data_import
import ModelTraining.datamodels.datamodels.validation.white_test
from ModelTraining.TrainingUtilities import export_metrics as metr_exp, training_utils as train_utils
from ModelTraining.Training.predict import predict_gt
from ModelTraining.Training.ModelCreation import create_model
from ModelTraining.Training.GridSearch import best_estimator
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults


if __name__ == '__main__':
    data_dir_path = "../"
    results_dir_path = "./results/"
    usecase_config_path = os.path.join("./", 'Configuration','UseCaseConfig')
    usecase_name = 'Beyond_B20_Gas'
    dict_usecase = data_import.load_from_json(os.path.join(usecase_config_path, f"{usecase_name}.json"))

    data, feature_set = data_import.get_data_and_feature_set(os.path.join(data_dir_path, dict_usecase['dataset']), os.path.join("./", dict_usecase['fmu_interface']))
    target_features = feature_set.get_output_feature_names()

    # Added: Preprocessing - Smooth features
    smoothe_data = False
    plot_enabled = True

    model_type = "RidgeRegression"

    training_params = TrainingParams(model_type=model_type,
                                       model_name="Energy",
                                       lookback_horizon=1,
                                       target_features=target_features,
                                       prediction_horizon=1,
                                       static_input_features=feature_set.get_static_feature_names(),
                                       dynamic_input_features=feature_set.get_dynamic_feature_names(),
                                       training_split=0.8,
                                       normalizer="Normalizer",
                                       expansion=['IdentityExpander'])

    # Preprocess data
    dp_utils.preprocess_data(data, feature_set, smoothe_data=smoothe_data)

    # Extract data and reshape
    index, x, y, feature_names = ModelTraining.TrainingUtilities.preprocessing.extract_training_and_test_set(data, training_params)
    index_train, x_train, y_train, index_test, x_test, y_test = ModelTraining.TrainingUtilities.preprocessing.split_into_training_and_test_set(
        index, x, y, training_params.training_split)

    # Create model
    logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")
    model = create_model(training_params, feature_names=feature_names)
    # Grid search
    best_params = best_estimator(model, x_train, y_train, parameters={})
    model.model.set_params(**best_params)
    # Train model
    model.train(x_train, y_train)
    # Save Model
    train_utils.save_model_and_parameters(os.path.join(results_dir_path, f"Models/{training_params.model_name}/{training_params.model_type}_{training_params.expansion[0]}"), model, training_params)
    # Predict test data
    result_prediction = predict_gt(model, index_test, x_test, y_test, training_params)

    # Calculate and export metrics
    results = []
    metrics_exp = MetricsExport(plot_enabled=plot_enabled, results_root=results_dir_path)
    for feature in training_params.target_features:
        y_true = result_prediction[feature].to_numpy()
        y_pred = result_prediction[f"predicted_{feature}"].to_numpy()
        # Calculate Metrics
        metrics = metr_exp.calc_metrics(y_true, y_pred, x.shape[0], len(model.get_expanded_feature_names()),
                                 metrics_names=['R2', 'CV-RMS', 'MAPE'])
        # White test
        #white_test_results = ModelTraining.datamodels.datamodels.validation.white_test.white_test(x_test, y_true - y_pred)

        test_prediction = result_prediction[[f"predicted_{feature}" for feature in target_features]].to_numpy()
        results.append(TrainingResults(train_index=index_train, train_target=y_train,
                                 test_index=index_test, test_target=y_test, test_prediction=test_prediction))
        # Export metrics
        metrics_exp.export_results(model, target_features, result_prediction)

    print('Experiment finished')