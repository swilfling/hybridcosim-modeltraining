import os
import logging
from sklearn.pipeline import make_pipeline
from ModelTraining.Preprocessing import data_preprocessing as dp_utils
from ModelTraining.feature_engineering.featurecreators import CyclicFeatures, CategoricalFeatures
from ModelTraining.dataimport.data_import import DataImport, load_from_json
import ModelTraining.datamodels.datamodels.validation.white_test
from ModelTraining.feature_engineering.featureset import FeatureSet
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.datamodels.datamodels import Model
from ModelTraining.feature_engineering.expandedmodel import TransformerSet, ExpandedModel
from ModelTraining.datamodels.datamodels.processing import DataScaler
from ModelTraining.Training.GridSearch import best_estimator, best_pipeline
from ModelTraining.feature_engineering.parameters import TrainingParams, TrainingParamsExpanded, TransformerParams
from ModelTraining.Utilities import TrainingData
from ModelTraining.Utilities.MetricsExport.metrics_calc import MetricsCalc
from ModelTraining.Utilities.MetricsExport.result_export import ResultExport
import ModelTraining.Utilities.MetricsExport.metr_utils as metr_utils
from ModelTraining.feature_engineering.filters import ButterworthFilter

if __name__ == '__main__':
    data_dir_path = "../"
    config_path = os.path.join("./", 'Configuration')
    usecase_name = 'Solarhouse1_T'
    result_dir = f"./results/{usecase_name}"
    os.makedirs(result_dir, exist_ok=True)
    dict_usecase = load_from_json(os.path.join(config_path, 'UseCaseConfig', f"{usecase_name}.json"))
    data_import = DataImport.load(os.path.join(config_path, "DataImport", f"{dict_usecase['dataset_filename']}.json"))
    data = data_import.import_data(os.path.join(data_dir_path, dict_usecase['dataset_dir'], dict_usecase['dataset_filename']))
    feature_set = FeatureSet(os.path.join("./", dict_usecase['fmu_interface']))
    #data = data[15000:21000]

    # Added: Preprocessing - Smooth features
    smoothe_data = True
    plot_enabled = True

    data = dp_utils.preprocess_data(data, filename=dict_usecase['dataset_filename'])

    # Cyclic, categorical and statistical features
    cyclic_feat_cr = CyclicFeatures(dict_usecase.get('cyclical_feats', []))
    categorical_feat_cr = CategoricalFeatures(dict_usecase.get('onehot_feats', []))
    preproc_steps = [cyclic_feat_cr, categorical_feat_cr]
    # Smoothing - filter
    if smoothe_data:
        preproc_steps.insert(0, ButterworthFilter(order=2, T=10, keep_nans=False, remove_offset=True,
                               features_to_transform=dict_usecase.get('to_smoothe',[])))

    preproc = make_pipeline(*preproc_steps, 'passthrough')
    data = preproc.fit_transform(data)

    #data.to_csv(os.path.join(result_dir, f'{usecase_name}_preprocessed.csv'),sep=";", index_label='datetime')
    feature_set.add_cyclic_input_features(cyclic_feat_cr.get_additional_feat_names() + categorical_feat_cr.get_additional_feat_names())
    target_features = feature_set.get_output_feature_names()

    model_type = "LinearRegression"

    training_params = TrainingParamsExpanded(model_type=model_type,
                                       model_name="Energy",
                                       lookback_horizon=5,
                                       target_features=target_features,
                                       prediction_horizon=1,
                                       static_input_features=feature_set.get_static_input_feature_names(),
                                       dynamic_input_features=feature_set.get_dynamic_input_feature_names(),
                                       dynamic_output_features=feature_set.get_dynamic_output_feature_names(),
                                       training_split=0.8,
                                       normalizer="Normalizer",
                                             transformer_params=[
                                                 TransformerParams('DynamicFeatures', {'lookback_horizon': 5,
                                                                                       'flatten_dynamic_feats': True,
                                                                                       'features_to_transform':[c in feature_set.get_dynamic_input_feature_names() for c in feature_set.get_input_feature_names()]}),
                                                 TransformerParams('RThreshold', {'thresh': 0.05})])


    # Extract data and reshape
    #index, x, y, feature_names = ModelTraining.Training.TrainingUtilities.training_utils.extract_training_and_test_set(data, training_params)
    index = data.index
    x = data[training_params.static_input_features + training_params.dynamic_input_features]
    y = data[training_params.target_features]
    feature_names = training_params.static_input_features + training_params.dynamic_input_features

    index_train, x_train, y_train, index_test, x_test, y_test = ModelTraining.Training.TrainingUtilities.training_utils.split_into_training_and_test_set(
        index, x, y, training_params.training_split)


    # Create model
    logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")
    model_basic = Model.from_name(training_params.model_type,
                                  x_scaler_class=DataScaler.cls_from_name(training_params.normalizer),
                                  name=training_params.str_target_feats(), parameters={})

    transformers = TransformerSet.from_list_params(training_params.transformer_params)
    model = ExpandedModel(transformers=transformers, model=model_basic, feature_names=feature_names)
    # Grid search
    parameters = {"rthreshold__thresh": [0.001, 0.01, 0.02, 0.05, 0.1],
                  "dynamicfeatures__lookback_horizon": [4,8,12, 24]}
    best_params = best_pipeline(model, x_train, y_train, parameters=parameters)
    #print(best_params)
    for k, val in best_params.items():
        model.transformers.set_transformer_attributes(k.split("__")[0], {k.split("__")[1]: val})
    #best_params = best_estimator(model, x_train, y_train, parameters={})
    #model.get_estimator().set_params(**best_params)
    # Train model
    model.train(x_train, y_train)
    transformers.get_transformer_by_index(1).print_metrics()
    # Save Model
    output_dir = os.path.join(result_dir, f"Models/{training_params.model_name}/{training_params.model_type}")
    train_utils.save_model_and_params(model, training_params, output_dir)
    # Predict test data
    prediction_length = 7
    #y_forecast = predict_history_ar(model, index_test, x_test, y_test, training_params, prediction_length=prediction_length, feature_names=feature_names)
   # result_forecast = TrainingResults(test_target=y_test[:prediction_length + training_params.lookback_horizon + 1],
   #                                   test_prediction=y_forecast,
   #                                   test_index=index_test[:prediction_length + training_params.lookback_horizon + 1],
   #                                   test_input=x_test[:prediction_length + training_params.lookback_horizon + 1],
   #                                   target_feat_names=target_features)
   # plt_utils.plot_data(result_forecast.test_result_df(), result_dir, "result_forecast")

    # Calculate and export metrics
    test_prediction = model.predict(x_test)
    result_data = TrainingData(train_index=index_train.to_numpy(),
                               train_target=y_train.to_numpy(),
                               test_index=index_test.to_numpy()[training_params.lookback_horizon:],
                               test_target=y_test.to_numpy()[training_params.lookback_horizon:],
                               test_prediction=test_prediction[training_params.lookback_horizon:],
                               test_input=x_test.to_numpy()[training_params.lookback_horizon:],
                               target_feat_names=target_features)
    result_data.save_pkl(result_dir, "result_data.pkl")
    # Calculate metrics
    metr_exp = MetricsCalc()
    df_metrics = metr_exp.calc_perf_metrics_df(result_data, df_index=[model_type])
    #df_white = metr_exp.white_test_df(result_data, df_index=[model_type])
    df_metrics.to_csv(os.path.join(result_dir, f"Metrics_{metr_utils.create_file_name_timestamp()}.csv"), index_label='Model',
                    float_format='%.3f')
    #df_white.to_csv(os.path.join(result_dir, f"White_{metr_utils.create_file_name_timestamp()}.csv"),
    #                  index_label='Model', float_format='%.3f')
    # Export results
    result_exp = ResultExport(results_root=result_dir, plot_enabled=True)
    result_exp.export_result(result_data)
    result_exp.export_featsel_metrs(model)
    result_exp.export_model_properties(model)
    print('Experiment finished')