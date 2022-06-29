import os
import logging
from sklearn.pipeline import make_pipeline
from ModelTraining.Preprocessing import data_preprocessing as dp_utils
from ModelTraining.dataimport.data_import import DataImport, load_from_json
from ModelTraining.feature_engineering.featureset import FeatureSet
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.datamodels.datamodels import Model
from ModelTraining.feature_engineering.expandedmodel import TransformerSet, ExpandedModel
from ModelTraining.feature_engineering.feature_selectors import FeatureSelector
from ModelTraining.datamodels.datamodels.processing import DataScaler
from ModelTraining.feature_engineering.parameters import TrainingParams, TrainingParamsExpanded, TransformerParams
from ModelTraining.Utilities.MetricsExport.metrics_calc import MetricsCalc, MetricsVal
from ModelTraining.Utilities.MetricsExport.result_export import ResultExport
import ModelTraining.Utilities.MetricsExport.metr_utils as metr_utils
from ModelTraining.feature_engineering.filters import ButterworthFilter
from sklearn.model_selection import GridSearchCV

if __name__ == '__main__':
    data_dir_path = "../"
    config_path = os.path.join("./", 'Configuration')
    usecase_name = 'Beyond_T24_dyn'
    experiment_name = metr_utils.create_file_name_timestamp()
    result_dir = f"./results/{experiment_name}"
    os.makedirs(result_dir, exist_ok=True)

    model_type = "RidgeRegression"
    model_types = ['LinearRegression', 'RidgeRegression', 'LassoRegression','RandomForestRegression']
    model_names = {'RidgeRegression': 'ridge', 'RandomForestRegression': 'randomforestregressor',
                   "LinearRegression": "linearregression", "LassoRegression": "lasso",
                   'RuleFitRegression': 'rulefit', 'XGBoost': 'xgboost'}

    transformer_type = 'RThreshold'
    transformer_params = [TransformerParams(type=transformer_type, params={'thresh': 0.05})]
    transformer_name = transformer_type.lower()
    thresh_params = {f"{transformer_name}__thresh": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]}
    gridsearch_scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    gridsearch_params = {model_type:load_from_json( os.path.join(config_path, "GridSearchParameters", f'parameters_{model_type}.json')) for model_type in model_types}

    training_params = TrainingParamsExpanded(model_name="Tint",
                                             training_split=0.8,
                                             normalizer="Normalizer",
                                             transformer_params=transformer_params)

    # Added: Preprocessing - Smooth features
    smoothe_data = True
    plot_enabled = True

    ############################### Preprocessing ######################################################################

    # Get main config
    dict_usecase = load_from_json(os.path.join(config_path, 'UseCaseConfig', f"{usecase_name}.json"))
    # Get data
    data_import = DataImport.load(os.path.join(config_path, "DataImport", f"{dict_usecase['dataset_filename']}.json"))
    data_filename = os.path.join(data_dir_path, dict_usecase['dataset_dir'], dict_usecase['dataset_filename'])
    data = data_import.import_data(data_filename)
    # Configure training params
    feature_set = FeatureSet(os.path.join("./", dict_usecase['fmu_interface']))
    training_params = train_utils.set_train_params_model(training_params, feature_set, "Tint", model_type,transformer_params)
    # Preporocessing
    data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], filename=dict_usecase['dataset_filename'], do_smoothe=False)
    # Smoothing - filter
    preproc_steps = []
    if smoothe_data:
        preproc_steps.insert(0, ButterworthFilter(order=3, T=5, keep_nans=False, remove_offset=True,
                               features_to_transform=[feature in dict_usecase['to_smoothe'] for feature in data.columns]))

    preproc = make_pipeline(*preproc_steps, 'passthrough')
    import pandas as pd
    data = pd.DataFrame(columns=data.columns, index=data.index, data=preproc.fit_transform(data))


    ####################################### Main loop #################################################################

    metr_exp = MetricsCalc()
    for model_type in model_types:
        for lookback_horizon in [4, 5, 8, 10, 15, 20]:
            training_params.model_type = model_type
            training_params.lookback_horizon = lookback_horizon
            result_dir_model = os.path.join(result_dir, f"{model_type}_{lookback_horizon}")
            os.makedirs(result_dir_model)

            # Extract data and reshape
            index, x, y, feature_names = train_utils.extract_training_and_test_set(data, training_params)
            train_data = train_utils.create_train_data(index, x, y, training_params.training_split)
            x_train, y_train = train_data.train_input, train_data.train_target
            train_data.target_feat_names = training_params.target_features
            # Create model
            logging.info(f"Training model with input of shape: {x_train.shape} "
                         f"and targets of shape {y_train.shape}")
            model_basic = Model.from_name(training_params.model_type,
                                          x_scaler_class=DataScaler.cls_from_name(training_params.normalizer),
                                          name=training_params.str_target_feats(), parameters={})
            transformers = TransformerSet.from_list_params(training_params.transformer_params)
            model = ExpandedModel(transformers=transformers, model=model_basic, feature_names=feature_names)
            # Grid search
            parameters = thresh_params.copy()
            for name, vals in gridsearch_params[training_params.model_type].items():
                parameters.update({f"{model_names[model_type]}__{name}":vals})
            search = GridSearchCV(model.get_full_pipeline(), parameters, cv=2, scoring=gridsearch_scoring, refit='r2',
                                  verbose=4)
            # Transform x train
            search.fit(*model.preprocess(x_train, y_train))
            print(f"Best score for model {model.__class__.__name__} - {model.model.__class__.__name__} is: {search.best_score_}")
            print(f"Best parameters are {search.best_params_}")
            for k, val in search.best_params_.items():
                if k.split("__")[0] == transformer_name:
                    model.transformers.set_transformer_attributes(k.split("__")[0], {k.split("__")[1]: val})
                    metr_exp.add_metr_val(MetricsVal(model_type=model_type,
                                                     model_name=model.name,
                                                     featsel_thresh=transformer_name,
                                                     expansion_type=str(lookback_horizon),
                                                     metrics_type="best_params", metrics_name=k, val=val, usecase_name=usecase_name))
                else:
                    setattr(model.get_estimator(), k.split("__")[1], val)

            # Train model
            model.train(x_train, y_train)
            model.transformers.get_transformer_by_name(transformer_name).print_metrics()
            # Save Model
            model_dir = os.path.join(result_dir_model, f"Models/{training_params.model_name}/{training_params.model_type}")
            train_utils.save_model_and_params(model, training_params, model_dir)
            # Calculate and export metrics
            train_data.test_prediction = model.predict(train_data.test_input)
            train_data.save_pkl(result_dir_model, "result_data.pkl")
            # Calc metrics
            metr_vals = metr_exp.calc_all_metrics(train_data, model.transformers.get_transformers_of_type(FeatureSelector))
            # Set metrics identifiers
            for metr_val in metr_vals:
                metr_val.set_metr_properties(model_type, model.name, str(lookback_horizon), transformer_name, usecase_name)
            # Store metrics separately
            metr_exp.add_metr_vals(metr_vals)
            for type in ['Metrics', 'pvalues', 'FeatureSelect']:
                filename = os.path.join(result_dir_model, f"{type}_{experiment_name}.csv")
                metr_exp.store_metr_df(metr_exp.get_metr_df(type), filename)
            # Export results
            ResultExport(results_root=result_dir_model, plot_enabled=True).export_result_full(model,train_data, str(lookback_horizon))
        # Store all metrics
        metr_exp.store_all_metrics(result_dir, index_col='expansion_type')
        print('Experiment finished')
