import os
import logging
from sklearn.pipeline import make_pipeline
from ModelTraining.Preprocessing import data_preprocessing as dp_utils
from ModelTraining.feature_engineering.featurecreators import CyclicFeatures, StatisticalFeatures, CategoricalFeatures
from ModelTraining.dataimport.data_import import DataImport, load_from_json
import ModelTraining.datamodels.datamodels.validation.white_test
from ModelTraining.feature_engineering.featureset import FeatureSet
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.datamodels.datamodels import Model
from ModelTraining.feature_engineering.expandedmodel import TransformerSet, ExpandedModel
from ModelTraining.datamodels.datamodels.processing import DataScaler
from ModelTraining.Training.GridSearch import best_estimator, best_pipeline
from ModelTraining.feature_engineering.parameters import TrainingParams, TrainingParamsExpanded, TransformerParams
from ModelTraining.Utilities import TrainingResults
from ModelTraining.Utilities.MetricsExport.metrics_calc import MetricsCalc
from ModelTraining.Utilities.MetricsExport.result_export import ResultExport
import ModelTraining.Utilities.MetricsExport.metr_utils as metr_utils
from ModelTraining.feature_engineering.filters import ButterworthFilter

if __name__ == '__main__':
    data_dir_path = "../"
    config_path = os.path.join("./", 'Configuration')
    usecase_name = 'CPS-Data'
    result_dir = f"./results/{usecase_name}"
    os.makedirs(result_dir, exist_ok=True)
    dict_usecase = load_from_json(os.path.join(config_path, 'UseCaseConfig', f"{usecase_name}.json"))
    data_import = DataImport.load(os.path.join(config_path, "DataImport", f"{dict_usecase['dataset_filename']}.json"))
    data = data_import.import_data(os.path.join(data_dir_path, dict_usecase['dataset_dir'], dict_usecase['dataset_filename']))
    feature_set = FeatureSet(os.path.join("./", dict_usecase['fmu_interface']))

    # Added: Preprocessing - Smooth features
    smoothe_data = False
    plot_enabled = True

    data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], filename=dict_usecase['dataset_filename'], do_smoothe=smoothe_data)

    # Cyclic, categorical and statistical features
    cyclic_feat_cr = CyclicFeatures(dict_usecase.get('cyclical_feats', []))
    categorical_feat_cr = CategoricalFeatures(dict_usecase.get('onehot_feats', []))
    statistical_feat_cr = StatisticalFeatures(selected_feats=dict_usecase.get('stat_feats', []), statistical_features=dict_usecase.get('stat_vals', []),
                                              window_size=dict_usecase.get('stat_ws', 1))
    preproc_steps = [cyclic_feat_cr, categorical_feat_cr, statistical_feat_cr]
    # Smoothing - filter
    if smoothe_data:
        preproc_steps.insert(0, ButterworthFilter(order=2, T=10, keep_nans=False, remove_offset=True,
                               features_to_transform=dict_usecase['to_smoothe']))

    preproc = make_pipeline(*preproc_steps, 'passthrough')
    data = preproc.fit_transform(data)

    feature_set.add_cyclic_input_features(cyclic_feat_cr.get_additional_feat_names() + categorical_feat_cr.get_additional_feat_names())
    feature_set.add_statistical_input_features(statistical_feat_cr.get_additional_feat_names())
    target_features = feature_set.get_output_feature_names()

    model_type = "RidgeRegression"

    expander_parameters = load_from_json(os.path.join(config_path,'expander_params_PolynomialExpansion.json' ))

    transformer_params = [TransformerParams(type='MICThreshold', params={'thresh': 0.05}),
                          TransformerParams(type='PolynomialExpansion', params=expander_parameters),
                          TransformerParams(type='RThreshold', params={'thresh': 0.05})]

    training_params = TrainingParamsExpanded(model_type=model_type, model_name="Energy",
                                       lookback_horizon=10,
                                       target_features=target_features,
                                       prediction_horizon=1,
                                       static_input_features=feature_set.get_static_feature_names(),
                                       dynamic_input_features=feature_set.get_dynamic_input_feature_names(),
                                       dynamic_output_features=feature_set.get_dynamic_output_feature_names(),
                                       training_split=0.8,
                                       normalizer="Normalizer",
                                    transformer_params=transformer_params)


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
    #sel = SelectorByName(**load_from_json(os.path.join(config_path, "TransformerParams", "params_SelectorByName.json")))
    #sel.feature_names_in_ = feature_names

    transformers = TransformerSet.from_list_params(training_params.transformer_params)
    model = ExpandedModel(transformers=transformers, model=model_basic, feature_names=feature_names)
    # Grid search
    parameters = {"rthreshold__thresh": [0.01, 0.02, 0.05, 0.1, 0.2, 0.3]}
    gridsearch_params = load_from_json(os.path.join(config_path,"GridSearchParameters", f'parameters_{model_type}.json'))
    #for name, vals in gridsearch_params.items():
    #    parameters.update({f"ridge__{name}":vals})
    print(parameters)

    best_params = best_pipeline(model, x_train, y_train, parameters=parameters)
    print(best_params)
    model.transformers.get_transformer_by_name("rthreshold").set_params(thresh=best_params['rthreshold__thresh'])
    #best_params = best_estimator(model, x_train, y_train, parameters={})
    #model.get_estimator().set_params(**best_params)
    # Train model
    model.train(x_train, y_train)
    model.transformers.get_transformer_by_name("rthreshold").print_metrics()
    # Save Model
    model_dir = os.path.join(result_dir, f"Models/{training_params.model_name}/{training_params.model_type}")
    train_utils.save_model_and_params(model, training_params, model_dir)
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
    result_exp.export_featsel_metrs(model)
    result_exp.export_model_properties(model)
    print('Experiment finished')