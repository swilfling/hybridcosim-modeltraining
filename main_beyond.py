import os
import logging
import shutil
from ModelTraining.Preprocessing import data_preprocessing as dp_utils
from ModelTraining.Data.DataImport.featureset.featureset import FeatureSet
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.datamodels import datamodels as datamodels
from ModelTraining.datamodels.datamodels.processing import datascaler as datascaler
from ModelTraining.datamodels.datamodels.wrappers.expandedmodel import TransformerSet, ExpandedModel, TransformerParams
from ModelTraining.datamodels.datamodels.processing.datascaler import Normalizer
from ModelTraining.Training.TrainingUtilities.trainingparams_expanded import TrainingParamsExpanded
from ModelTraining.Utilities.MetricsExport.metrics_calc import MetricsCalc
import ModelTraining.Utilities.MetricsExport.metr_utils as metr_utils
from sklearn.model_selection import GridSearchCV


if __name__ == '__main__':
    data_dir_path = "./Data"
    config_path = os.path.join("./", 'Configuration')
    usecase_name = 'Beyond_B20_LR_dyn'
    experiment_name = metr_utils.create_file_name_timestamp()
    result_dir = f"./results/{experiment_name}"
    os.makedirs(result_dir, exist_ok=True)

    # Get main config
    usecase_config_file = os.path.join(config_path, 'UseCaseConfig', f"{usecase_name}.json")
    dict_usecase = train_utils.load_from_json(usecase_config_file)

    interface_file = os.path.join("./Data/Configuration/FeatureSet", dict_usecase['fmu_interface'])
    shutil.copy(usecase_config_file, os.path.join(result_dir, f"{usecase_name}.json"))
    shutil.copy(interface_file, os.path.join(result_dir, "feature_set.csv"))

    # Get feature set
    feature_set = FeatureSet(os.path.join("Data", "Configuration", "FeatureSet", dict_usecase['fmu_interface']))

    transformer_type = 'RThreshold'
    stat_feats = dict_usecase['stat_feats']
    stat_vals = dict_usecase['stat_vals']
    stat_ws = dict_usecase['stat_ws']

    transformer_params = TransformerParams.load_parameters_list("./Configuration/TransformerParams/params_transformers_beyond_stat_cyc_categ_inv_dyn_poly_r.json")
    transformer_name = transformer_type.lower()
    transf_params = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    #transf_params = [0.15]
    thresh_params = {f"{transformer_name}__thresh": transf_params}
    gridsearch_scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    model_type = 'LassoRegression'
    model_type = 'SupportVectorRegression'
    model_types = ['NeuralNetwork_sklearn','LassoRegression','RidgeRegression','SupportVectorRegression']

    #gridsearch_params = {model_type:train_utils.load_from_json(os.path.join(config_path, "GridSearchParameters", f'parameters_{model_type}.json')) for model_type in model_types}
    gridsearch_params = {model_type:{} for model_type in model_types}

    training_params = TrainingParamsExpanded(model_name="Tint",
                                             training_split=0.8,
                                             normalizer="Normalizer",
                                             transformer_params=transformer_params)

    ############################### Preprocessing ######################################################################

    # Get data
    dataimport_cfg_path = os.path.join(data_dir_path,"Configuration","DataImport")
    data = train_utils.import_data(dataimport_cfg_path,data_dir_path, dict_usecase)
    data = dp_utils.preprocess_data(data, filename=dict_usecase['dataset_filename'])
    training_params = train_utils.set_train_params_model(training_params, feature_set, feature_set.get_output_feature_names()[0], model_types[0],transformer_params)

    ####################################### Main loop ##################################################################
    training_data = data[training_params.static_input_features + training_params.dynamic_input_features]
    target_data = data[training_params.target_features]
    from ModelTraining.feature_engineering.featureengineering.filters import ButterworthFilter
    target_data = ButterworthFilter(remove_offset=True).fit_transform(target_data)
    target_data = Normalizer().fit(target_data).transform(target_data)
    lookbacks = [2,4,12,24]
    gridsearch_params['LassoRegression']= {'alpha':[0.01,0.1, 0.5,1,2,5,10,50,200]}
    transformers_mask = TransformerParams.get_params_of_type(transformer_params,"Transformer_MaskFeats")

    dynfeats_params = [params for params in transformers_mask if params.params['transformer_type'] == 'DynamicFeatures'][0]

    metr_exp = MetricsCalc()
    for model_type in model_types:
        for lookback_horizon in lookbacks:
            training_params.model_type = model_type
            training_params.lookback_horizon = lookback_horizon
            dynfeats_params.params['transformer_params'].update({'lookback_horizon':lookback_horizon})
            result_dir_model = os.path.join(result_dir, f"{model_type}_{lookback_horizon}")
            os.makedirs(result_dir_model)
            x, y, index, feature_names = training_data, target_data.to_numpy(), training_data.index, training_data.columns
            # structure
            train_data = train_utils.create_train_data(index, x, y, training_params.training_split)
            x_train, y_train = train_data.train_input, train_data.train_target
            train_data.target_feat_names = training_params.target_features
            # Create model
            logging.info(f"Training model with input of shape: {x_train.shape} "
                         f"and targets of shape {y_train.shape}")
            model_basic = getattr(datamodels, training_params.model_type)(
                                          x_scaler_class=getattr(datascaler,training_params.normalizer),
                                          name=training_params.str_target_feats(), parameters={})
            transformers = TransformerSet.from_list_params(training_params.transformer_params)

            #transformers.get_transformer_by_name('selectorbyname').selected_feat_names = train_utils.load_from_json("./Configuration/cfg_selectorbyname.json")
            model = ExpandedModel(transformers=transformers, model=model_basic, feature_names=feature_names)
            pipeline = model.get_full_pipeline()
            from sklearn.pipeline import make_pipeline
            from ModelTraining.feature_engineering.featureengineering.resamplers import SampleCut_Transf
            pipeline = make_pipeline(*[step[1] for step in pipeline.steps[:-2]],SampleCut_Transf(lookback_horizon),pipeline.steps[-1][1])
            #data_out = pipeline.fit_transform(x_train, y_train)
            #feat_names_full = pipeline.get_feature_names_out(x_train.columns)
            #print(len(feat_names_full))
            #print(len(transformers.get_transformer_by_name('selectorbyname').selected_feat_names))
            #transformers.get_transformer_by_name('selectorbyname').feature_names_in = feat_names_full
            # Grid search
            parameters = thresh_params.copy()
            #parameters = {}
            #for name, vals in gridsearch_params[training_params.model_type].items():
            #    parameters.update({f"{model.model.model.__class__.__name__.lower()}__{name}": vals})
            from sklearn.compose import TransformedTargetRegressor
            #from ModelTraining.feature_engineering.featureengineering.resamplers import SampleCut_Transf
            #reg = TransformedTargetRegressor(regressor=model.get_full_pipeline(), transformer=SampleCut_Transf(num_samples=lookback_horizon), check_inverse=False)
            search = GridSearchCV(model.get_full_pipeline(), parameters, cv=3, scoring=gridsearch_scoring, refit='r2',
                                  verbose=4)
            # Transform x train
            search.fit(*model.scale(x_train, y_train))
            print(f"Best score for model {model.__class__.__name__} - {model.model.model_type} is: {search.best_score_}")
            print(f"Best parameters are {search.best_params_}")
            for k, val in search.best_params_.items():
         #       if k.split("__")[0] == transformer_name:
         #           model.transformers.set_transformer_attributes(k.split("__")[0], {k.split("__")[1]: val})
         #       else:
                    setattr(model.get_estimator(), k.split("__")[1], val)

            # Train model
            model.train(x_train, y_train)
#            model.transformers.get_transformer_by_name(transformer_name).print_metrics()
            train_data.test_prediction = model.predict(train_data.test_input)
            # Save Model and results
            train_utils.store_results(model, train_data, training_params, metr_exp, search.best_params_, result_dir_model, str(lookback_horizon), experiment_name,
                          usecase_name)
        # Store all metrics
        metr_exp.store_all_metrics(result_dir, index_col='expansion_type')
        print('Experiment finished')
