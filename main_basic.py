import os
import logging
import ModelTraining.Preprocessing.FeatureCreation.add_features as feat_utils
from ModelTraining.Utilities.MetricsExport import metr_utils as metr_utils
import ModelTraining.Preprocessing.DataPreprocessing.data_preprocessing as dp_utils
import ModelTraining.Preprocessing.DataImport.data_import as data_import
import ModelTraining.datamodels.datamodels.validation.white_test
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.Training.predict import predict_history_ar
from ModelTraining.Training.ModelCreation import create_model
from ModelTraining.Training.GridSearch import best_estimator
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.Utilities.MetricsExport.MetricsCalc import MetricsCalc
from ModelTraining.Utilities.MetricsExport.ResultExport import ResultExport
from ModelTraining.Preprocessing.FeatureSelection import SelectorByName
import ModelTraining.Utilities.Plotting.plotting_utilities as plt_utils

if __name__ == '__main__':
    data_dir_path = "../"
    usecase_config_path = os.path.join("./", 'Configuration','UseCaseConfig')
    usecase_name = 'Beyond_T24_arx'
    result_dir = f"./results/{usecase_name}"
    os.makedirs(result_dir, exist_ok=True)
    dict_usecase = data_import.load_from_json(os.path.join(usecase_config_path, f"{usecase_name}.json"))

    data, feature_set = ModelTraining.Preprocessing.get_data_and_feature_set.get_data_and_feature_set(os.path.join(data_dir_path, dict_usecase['dataset']), os.path.join("./", dict_usecase['fmu_interface']))
    data, feature_set = feat_utils.add_features(data, feature_set, dict_usecase)
    target_features = feature_set.get_output_feature_names()

    # Added: Preprocessing - Smooth features
    smoothe_data = False
    plot_enabled = True

    model_type = "RandomForestRegression"

    training_params = TrainingParams(model_type=model_type,
                                       model_name="Energy",
                                       lookback_horizon=5,
                                       target_features=target_features,
                                       prediction_horizon=1,
                                       static_input_features=feature_set.get_static_feature_names(),
                                       dynamic_input_features=feature_set.get_dynamic_input_feature_names(),
                                       dynamic_output_features=feature_set.get_dynamic_output_feature_names(),
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

    list_sel_feat_names = ['Tint_1','Tint_2','Tint_3','Tint_5',
                           'Text','Text_1','Text_3', 'Text_4',
                           'GHI','GHI_1','GHI_2','GHI_4']
    selector = SelectorByName(feat_names=feature_names, selected_feat_names=list_sel_feat_names)
    model.expanders[0].set_feature_select(selector.get_support())
    print(feature_names)
    print("Support:")
    print(model.expanders[0].selected_features)

    # Grid search
    best_params = best_estimator(model, x_train, y_train, parameters={})
    model.model.set_params(**best_params)
    # Train model
    model.train(x_train, y_train)
    # Save Model
    train_utils.save_model_and_params(os.path.join(result_dir, f"Models/{training_params.model_name}/{training_params.model_type}_{training_params.expansion[-1]}"), model, training_params)
    # Predict test data
    prediction_length = 72
    y_forecast = predict_history_ar(model, index_test, x_test, y_test, training_params, prediction_length=prediction_length, feature_names=feature_names)
    result_forecast = TrainingResults(test_target=y_test[:prediction_length + training_params.lookback_horizon + 1],
                                      test_prediction=y_forecast,
                                      test_index=index_test[:prediction_length + training_params.lookback_horizon + 1],
                                      test_input=x_test[:prediction_length + training_params.lookback_horizon + 1],
                                      target_feat_names=target_features)
    plt_utils.plot_result(result_forecast.test_result_df(), result_dir, "result_forecast")

    # Calculate and export metrics
    test_prediction = model.predict(x_test)
    result_data = TrainingResults(train_index=index_train.to_numpy(), train_target=y_train, test_index=index_test.to_numpy(),
                                  test_target=y_test, test_prediction=test_prediction, test_input=x_test, target_feat_names=target_features)
    result_data.save_pkl(result_dir, "result_data.pkl")
    # Calculate metrics
    metr_exp = MetricsCalc()
    df_metrics = metr_exp.analyze_result(model, result_data)
    metr_exp.store_all_metrics(df_metrics, results_path=result_dir, timestamp=metr_utils.create_file_name_timestamp())
    # Export results
    result_exp = ResultExport(results_root=result_dir)
    result_exp.export_result(result_data)
    result_exp.export_model_properties(model)
    print('Experiment finished')