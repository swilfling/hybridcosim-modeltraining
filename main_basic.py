import os
import logging
import ModelTraining.Preprocessing.FeatureCreation.add_features as feat_utils
from ModelTraining.Utilities.MetricsExport import export_metrics as metr_exp
import ModelTraining.Preprocessing.DataPreprocessing.data_preprocessing as dp_utils
import ModelTraining.Preprocessing.DataImport.data_import as data_import
import ModelTraining.datamodels.datamodels.validation.white_test
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.Training.predict import predict_gt, predict_history_ar
from ModelTraining.Training.ModelCreation import create_model
from ModelTraining.Training.GridSearch import best_estimator
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.Utilities.MetricsExport.MetricsExport import analyze_result

if __name__ == '__main__':
    data_dir_path = "../"
    usecase_config_path = os.path.join("./", 'Configuration','UseCaseConfig')
    usecase_name = 'Beyond_T24_dyn'
    results_dir_path = f"./results/{usecase_name}"
    os.makedirs(results_dir_path, exist_ok=True)
    dict_usecase = data_import.load_from_json(os.path.join(usecase_config_path, f"{usecase_name}.json"))

    data, feature_set = ModelTraining.Preprocessing.get_data_and_feature_set.get_data_and_feature_set(os.path.join(data_dir_path, dict_usecase['dataset']), os.path.join("./", dict_usecase['fmu_interface']))
    data, feature_set = feat_utils.add_features(data, feature_set, dict_usecase)
    target_features = feature_set.get_output_feature_names()

    # Added: Preprocessing - Smooth features
    smoothe_data = False
    plot_enabled = True

    model_type = "LinearRegression"

    training_params = TrainingParams(model_type=model_type,
                                       model_name="Energy",
                                       lookback_horizon=4,
                                       target_features=target_features,
                                       prediction_horizon=1,
                                       static_input_features=feature_set.get_static_feature_names(),
                                       dynamic_input_features=feature_set.get_dynamic_feature_names(),
                                       training_split=0.8,
                                       normalizer="IdentityScaler",
                                       expansion=['IdentityExpander'])

    # Preprocess data
    data = dp_utils.preprocess_data(data, dict_usecase["to_smoothe"], do_smoothe=smoothe_data)

    # Extract data and reshape
    index, x, y, feature_names = ModelTraining.Training.TrainingUtilities.training_utils.extract_training_and_test_set(data, training_params)
    index_train, x_train, y_train, index_test, x_test, y_test = ModelTraining.Training.TrainingUtilities.training_utils.split_into_training_and_test_set(
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
    train_utils.save_model_and_parameters(os.path.join(results_dir_path, f"Models/{training_params.model_name}/{training_params.model_type}_{training_params.expansion[-1]}"), model, training_params)
    # Predict test data
    result_prediction = predict_gt(model, index_test, x_test, y_test, training_params)
    #result_prediction = predict_history_ar(model, index_test, x_test, y_test, training_params)
    # Calculate and export metrics
    test_prediction = result_prediction[[f"predicted_{feature}" for feature in target_features]].to_numpy()
    result_data = TrainingResults(train_index=index_train.to_numpy(), train_target=y_train,
                             test_index=index_test.to_numpy(), test_target=y_test, test_prediction=test_prediction, test_input=x_test)
    result_data.save_pkl(results_dir_path, "result_data.pkl")
    # Export metrics
    df_metrics = analyze_result([model], [result_data], [training_params], plot_enabled=plot_enabled,
                                results_dir_path=results_dir_path)
    metr_exp.store_all_metrics(df_metrics, results_path=results_dir_path, timestamp=metr_exp.create_file_name_timestamp())

    print('Experiment finished')