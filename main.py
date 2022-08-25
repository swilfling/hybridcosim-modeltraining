import os
import datetime
from ModelTraining.Data.DataImport.featureset.featureset import FeatureSet
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.Training.TrainingUtilities.parameters import TrainingParams
import ModelTraining.Data.Plotting.plot_data as plt_utils
from ModelTraining.Training.run_training_and_test import run_training_and_test
from ModelTraining.Data.DataImport.dataimport import DataImport

if __name__ == '__main__':

    data_dir = "AEE/Solarhouse1"
    data_filename = "Resampled15min"
    data_import = DataImport.load(os.path.join("Data", "Configuration", "DataImport", data_dir, f"{data_filename}.json"))
    data = data_import.import_data(os.path.join("Data","Data",data_dir, f'{data_filename}_proc'))

    feature_set_path = "Data/Configuration/FeatureSet/Solarhouse1/SolarCollector/FMUInterface_TSolar.csv"
    feature_set = FeatureSet(feature_set_path)
    # Get training and target features
    target_features = feature_set.get_output_feature_names()

    train_params_cfg = TrainingParams.load("Configuration/TrainingParameters/training_params.json")
    list_train_params = [train_params_cfg for feat in target_features]
    for feat, params in zip(target_features, list_train_params):
        params.model_name = feat
        params.dynamic_input_features = feature_set.get_dynamic_input_feature_names()
        params.static_input_features = feature_set.get_static_input_feature_names()
        params.dynamic_output_features = feature_set.get_dynamic_output_feature_names()
        params.target_features = [feat]

    print("Starting Training")

    training_results_path = f"results/Experiment_{datetime.datetime.now().strftime('%y%m%d_%H%M%S')}"
    os.makedirs(training_results_path, exist_ok=True)

    predict_type = "ground_truth"

    models, results = run_training_and_test(data, list_train_params,predict_type)

    print("Storing results")
    for params, result in zip(list_train_params, results):
        result_id = f'{params.model_type}_{params.model_name}'
        result.test_results_to_csv(training_results_path, f'{result_id}.csv')
        plt_utils.plot_data(result.test_result_df(),training_results_path,result_id)

    for params, model in zip(list_train_params, models):
        train_utils.save_model_and_params(model, params, training_results_path)

    print('Experiment finished')