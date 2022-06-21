import os
import logging
from sklearn.pipeline import make_pipeline
from ModelTraining.Preprocessing.FeatureCreation.featurecreators import CyclicFeatures, StatisticalFeatures, CategoricalFeatures
import ModelTraining.Preprocessing.DataImport.data_import as data_import
import ModelTraining.datamodels.datamodels.validation.white_test
from ModelTraining.Preprocessing.featureset import FeatureSet
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.Training.predict import predict_history_ar
from ModelTraining.datamodels.datamodels import Model
from ModelTraining.datamodels.datamodels.wrappers.feature_extension import TransformerSet, ExpandedModel, FeatureExpansion
from ModelTraining.datamodels.datamodels.processing import DataScaler
from ModelTraining.Training.GridSearch import best_estimator
from ModelTraining.Utilities.Plotting import plot_data as plt_utils
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.Utilities.MetricsExport.metrics_calc import MetricsCalc
from ModelTraining.Utilities.MetricsExport.result_export import ResultExport
from ModelTraining.Preprocessing.FeatureSelection.feature_selectors import SelectorByName
import ModelTraining.Utilities.MetricsExport.metr_utils as metr_utils
from ModelTraining.Preprocessing.DataPreprocessing.filters import ButterworthFilter

if __name__ == '__main__':
    data_dir_path = "../"
    usecase_config_path = os.path.join("./", 'Configuration','UseCaseConfig')
    usecase_name = 'Beyond_T24_arx'
    result_dir = f"./results/{usecase_name}"
    os.makedirs(result_dir, exist_ok=True)
    dict_usecase = data_import.load_from_json(os.path.join(usecase_config_path, f"{usecase_name}.json"))

    data = ModelTraining.Preprocessing.get_data_and_feature_set.get_data(
        os.path.join(data_dir_path, dict_usecase['dataset']))

    feature_set = FeatureSet(os.path.join("./", dict_usecase['fmu_interface']))

    # Added: Preprocessing - Smooth features
    smoothe_data = True
    plot_enabled = True

    # Cyclic, categorical and statistical features
    cyclic_feat_cr = CyclicFeatures(dict_usecase.get('cyclical_feats', []))
    categorical_feat_cr = CategoricalFeatures(dict_usecase.get('onehot_feats', []))
    statistical_feat_cr = StatisticalFeatures(dict_usecase.get('stat_feats', []), dict_usecase.get('stat_vals', []),
                                              dict_usecase.get('stat_ws', 1))
    preproc_steps = [cyclic_feat_cr, categorical_feat_cr, statistical_feat_cr]
    # Smoothing - filter
    if smoothe_data:
        preproc_steps.insert(0, ButterworthFilter(order=2, T=10, keep_nans=False, remove_offset=True,
                               features_to_filter=dict_usecase['to_smoothe']))

    preproc = make_pipeline(*preproc_steps, 'passthrough')
    data = preproc.fit_transform(data)

    feature_set.add_cyclic_input_features(cyclic_feat_cr.get_additional_feat_names() + categorical_feat_cr.get_additional_feat_names())
    feature_set.add_statistical_input_features(statistical_feat_cr.get_additional_feat_names())
    target_features = feature_set.get_output_feature_names()

    model_type = "LinearRegression"

    training_params = TrainingParams(model_type=model_type,
                                       model_name="Energy",
                                       lookback_horizon=5,
                                       target_features=target_features,
                                       prediction_horizon=1,
                                       static_input_features=feature_set.get_static_feature_names(),
                                       dynamic_input_features=feature_set.get_dynamic_input_feature_names(),
                                       dynamic_output_features=feature_set.get_dynamic_output_feature_names(),
                                       training_split=0.8,
                                       normalizer="Normalizer",
                                       expansion=['IdentityExpander'])


    # Extract data and reshape
    index, x, y, feature_names = ModelTraining.Training.TrainingUtilities.training_utils.extract_training_and_test_set(data, training_params)
    index_train, x_train, y_train, index_test, x_test, y_test = ModelTraining.Training.TrainingUtilities.training_utils.split_into_training_and_test_set(
        index, x, y, training_params.training_split)

    # Create model
    logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")
    model_basic = Model.from_name(training_params.model_type,
                                  x_scaler_class=DataScaler.cls_from_name(training_params.normalizer),
                                  name=training_params.str_target_feats(), parameters={})

    # Create expanded model
    expanders = FeatureExpansion.from_names(training_params.expansion)
    list_sel_feat_names = ['Tint_1', 'Tint_2', 'Tint_3', 'Tint_5',
                           'Text', 'Text_1', 'Text_3', 'Text_4',
                           'GHI', 'GHI_1', 'GHI_2', 'GHI_4']
    selector = SelectorByName(feat_names=feature_names, selected_feat_names=list_sel_feat_names)
    transformers = TransformerSet(expanders + [selector])
    model = ExpandedModel(transformers=transformers, model=model_basic)
    # Grid search
    best_params = best_estimator(model, x_train, y_train, parameters={})
    model.get_estimator().set_params(**best_params)
    # Train model
    model.train(x_train, y_train)
    model.transformers.get_transformer_by_name('selectorbyname').print_metrics()
    # Save Model
    train_utils.save_model_and_params(model, training_params, os.path.join(result_dir,
                                                                           f"Models/{training_params.model_name}/{training_params.model_type}_{training_params.expansion[-1]}"))
    # Predict test data
    prediction_length = 72
    y_forecast = predict_history_ar(model, index_test, x_test, y_test, training_params, prediction_length=prediction_length, feature_names=feature_names)
    result_forecast = TrainingResults(test_target=y_test[:prediction_length + training_params.lookback_horizon + 1],
                                      test_prediction=y_forecast,
                                      test_index=index_test[:prediction_length + training_params.lookback_horizon + 1],
                                      test_input=x_test[:prediction_length + training_params.lookback_horizon + 1],
                                      target_feat_names=target_features)
    plt_utils.plot_data(result_forecast.test_result_df(), result_dir, "result_forecast")

    # Calculate and export metrics
    test_prediction = model.predict(x_test)
    result_data = TrainingResults(train_index=index_train.to_numpy(), train_target=y_train, test_index=index_test.to_numpy(),
                                  test_target=y_test, test_prediction=test_prediction, test_input=x_test, target_feat_names=target_features)
    result_data.save_pkl(result_dir, "result_data.pkl")
    # Calculate metrics
    metr_exp = MetricsCalc()
    df_metrics = metr_exp.calc_perf_metrics_df(result_data, df_index=[model_type])
    df_white = metr_exp.white_test_df(result_data, df_index=[model_type])
    df_metrics.to_csv(os.path.join(result_dir, f"Metrics_{metr_utils.create_file_name_timestamp()}.csv"), index_label='Model',
                    float_format='%.3f')
    df_white.to_csv(os.path.join(result_dir, f"White_{metr_utils.create_file_name_timestamp()}.csv"),
                      index_label='Model', float_format='%.3f')
    # Export results
    result_exp = ResultExport(results_root=result_dir, plot_enabled=True)
    result_exp.export_result(result_data)
    result_exp.export_model_properties(model)
    print('Experiment finished')