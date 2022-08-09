import os
import logging
import shutil
from ModelTraining.Preprocessing import data_preprocessing as dp_utils
from ModelTraining.dataimport.data_import import DataImport, load_from_json
from DataImport.featureset import FeatureSet
from ModelTraining.Training.TrainingUtilities import training_utils as train_utils
from ModelTraining.datamodels import datamodels
from ModelTraining.datamodels.datamodels.processing import datascaler
from ModelTraining.datamodels.datamodels.wrappers.expandedmodel import TransformerSet, TransformerParams, ExpandedModel
from ModelTraining.feature_engineering.featureengineering.featureselectors import FeatureSelector
from ModelTraining.datamodels.datamodels.processing import Normalizer
from ModelTraining.feature_engineering.parameters import TrainingParamsExpanded
from ModelTraining.Utilities.MetricsExport.metrics_calc import MetricsCalc, MetricsVal
from ModelTraining.Utilities.MetricsExport.result_export import ResultExport
import ModelTraining.Utilities.MetricsExport.metr_utils as metr_utils
from sklearn.model_selection import GridSearchCV
from ModelTraining.feature_engineering.experimental.dynamicfeaturessamplecut import DynamicFeaturesSampleCut
from ModelTraining.feature_engineering.featureengineering.compositetransformers import Transformer_MaskFeats


if __name__ == '__main__':
    data_dir_path = "../Data"
    config_path = os.path.join("./", 'Configuration')
    beyond_building = "B20"
    usecase_name = f'Beyond_{beyond_building}_LR_dyn'
    experiment_name = metr_utils.create_file_name_timestamp()
    result_dir = f"./results/{experiment_name}"
    os.makedirs(result_dir, exist_ok=True)

    model_type = "RidgeRegression"
    model_types = [model_type]
    #model_types = ['RidgeRegression', 'RandomForestRegression']

    # Get main config
    usecase_config_file = os.path.join(config_path, 'UseCaseConfig', f"{usecase_name}.json")
    dict_usecase = load_from_json(usecase_config_file)

    interface_file = os.path.join("./", dict_usecase['fmu_interface'])
    shutil.copy(usecase_config_file, os.path.join(result_dir, f"{usecase_name}.json"))
    shutil.copy(interface_file, os.path.join(result_dir, "feature_set.csv"))

    # Get feature set
    feature_set = FeatureSet(interface_file)
    feats_to_invert = dict_usecase.get('to_invert', [])
    transformer_type = 'RThreshold'
    stat_feats = dict_usecase['stat_feats']
    stat_vals = dict_usecase['stat_vals']
    stat_ws = dict_usecase['stat_ws']

    inv_params = TransformerParams(type='Transformer_MaskFeats', params={
            'transformer_type':'InverseTransform',
            'mask_type': 'MaskFeats_Inplace',
            'mask_params':{'features_to_transform':feats_to_invert, 'rename_df_cols':False}})

    transformer_params = TransformerParams.load_parameters_list("Configuration/TransformerParams/params_transformers_beyond_stat_cyc_categ_poly_r.json")

    transformer_name = transformer_type.lower()
    transf_params = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    transf_params = [0.01, 0.1, 0.2, 0.3,0.45, 0.5]
    transf_params = [0.45]
    thresh_params = {f"{transformer_name}__thresh": transf_params}

    gridsearch_scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
    gridsearch_params = {model_type:load_from_json(os.path.join(config_path, "GridSearchParameters", f'parameters_{model_type}.json')) for model_type in model_types}

    training_params = TrainingParamsExpanded(model_name="Tint",
                                             training_split=0.8,
                                             normalizer="Normalizer",
                                             transformer_params=transformer_params)

    # Added: Preprocessing - Smooth features
    smoothe_data = True
    plot_enabled = True

    ############################### Preprocessing ######################################################################

    # Get data
    data_import = DataImport.load(os.path.join(config_path, "DataImport", f"{dict_usecase['dataset_filename']}.json"))
    data_filename = os.path.join(data_dir_path, dict_usecase['dataset_dir'], dict_usecase['dataset_filename'])
    data = data_import.import_data(data_filename)
    # Preporocessing
    data = dp_utils.preprocess_data(data, filename=dict_usecase['dataset_filename'])

    # Configure training params
    training_params = train_utils.set_train_params_model(training_params, feature_set, feature_set.get_output_feature_names()[0], model_type,transformer_params)

    ####################################### Main loop ##################################################################
    training_data = data[training_params.static_input_features + training_params.dynamic_input_features]
    target_data = data[training_params.target_features]
    target_data = Normalizer().fit(target_data).transform(target_data)

    #highpass = ButterworthFilter(filter_type='highpass', T=np.array([24]), order=3, remove_offset=True)
    #lowpass = ButterworthFilter(filter_type='lowpass', T=np.array([5]), order=2, remove_offset=True)
    #target_data = pd.DataFrame(data=lowpass.fit_transform(target_data[training_params.target_features[0]]),
    #                           index=target_data.index)
    #env = Envelope_MA(T=24*7)
    #target_data = env.fit_transform(target_data)
    #data_full = training_data = training_data.join(target_data).dropna()
    #training_data = data_full[training_params.static_input_features + training_params.dynamic_input_features]
    #target_data = data_full[training_params.target_features]

    lookbacks = [2,4,8,12,24]
    gridsearch_params['RidgeRegression'] = {'alpha': [10,20,50,100,200,500,1000]}

    #gridsearch_params['RidgeRegression'] = {'alpha': [10000]}
#    gridsearch_params['RidgeRegression'] = {'alpha': [10]}
    gridsearch_params['RandomForestRegression'] = {}
    gridsearch_params['LassoRegression'] = {}
    #thresh_params[f'{transformer_name}__thresh'] = [0.05,0.1,0.15,0.2]
    #thresh_params[f'{transformer_name}__thresh'] = [0]
    model_types = ['RidgeRegression','RandomForestRegression']
    model_types = ['RidgeRegression','LassoRegression']
    metr_exp = MetricsCalc()

    for lookback_horizon in lookbacks:
        training_params.lookback_horizon = lookback_horizon
        metr_label = str(lookback_horizon)

        # Extract data and reshape
        #micthresh = MICThreshold(thresh=0.005, omit_zero_samples=False)
        #training_data_thresh = micthresh.fit_transform(training_data, target_data)
        #feat_names_thresh = micthresh.get_feature_names_out(training_data.columns)
        training_data_thresh = training_data
        feat_names_thresh = training_data.columns

        tr = Transformer_MaskFeats(**inv_params.params)
        training_data_thresh = tr.fit_transform(training_data_thresh)

        if len(training_params.dynamic_input_features) > 0:
            dynfeats = DynamicFeaturesSampleCut(
                features_to_transform=[feat in training_params.dynamic_input_features for feat in feat_names_thresh],
                lookback_horizon=training_params.lookback_horizon,
                flatten_dynamic_feats=True)
            x, y = dynfeats.fit_resample(training_data_thresh, target_data.to_numpy())
            index = data.index[dynfeats.lookback_horizon:]
            feature_names = dynfeats.get_feature_names_out(feat_names_thresh)
        else:
            x, y, index, feature_names = training_data_thresh, target_data.to_numpy(), training_data.index, feat_names_thresh

        for model_type in model_types:
            result_dir_model = os.path.join(result_dir, f"{model_type}_{lookback_horizon}")
            os.makedirs(result_dir_model)
            # structure
            train_data = train_utils.create_train_data(index, x, y, training_params.training_split)
            x_train, y_train = train_data.train_input, train_data.train_target
            train_data.target_feat_names = training_params.target_features

           # from ModelTraining.feature_engineering.featureengineering.sampleweight import SampleWeight_DBSCAN
          #  sampleweight = SampleWeight_DBSCAN(weight_outlier_samples=0.8, weight_core_samples=1)
          #  sampleweight.fit(x_train, y_train)
          #  additional_params = {'sample_weight': sampleweight.weights_}
          #  print(sampleweight.get_num_core_samples())
          #  print(x_train.shape[0])
           # Create model
            logging.info(f"Training model with input of shape: {x_train.shape} "
                         f"and targets of shape {y_train.shape}")
            model_basic = getattr(datamodels, training_params.model_type)(
                                          x_scaler_class=getattr(datascaler, training_params.normalizer),
                                          name=training_params.str_target_feats(), parameters={})
            transformers = TransformerSet.from_list_params(training_params.transformer_params)
            model = ExpandedModel(transformers=transformers, model=model_basic, feature_names=feature_names)
            # Grid search
            parameters = thresh_params.copy()
            for name, vals in gridsearch_params[training_params.model_type].items():
                parameters.update({f"{model.model.model.__class__.__name__.lower()}__{name}": vals})
            import json
            with open(os.path.join(result_dir, f'gridsearch_params_{model_type}.json'), "w") as f:
                json.dump(gridsearch_params, f)
            search = GridSearchCV(model.get_full_pipeline(), parameters, cv=5, scoring=gridsearch_scoring, refit='r2',
                                  verbose=4)
            # Transform x train
            search.fit(*model.scale(x_train, y_train))
            print(f"Best score for model {model.__class__.__name__} - {model.model.__class__.__name__} is: {search.best_score_}")
            print(f"Best parameters are {search.best_params_}")
            for k, val in search.best_params_.items():
                if k.split("__")[0] == transformer_name:
                    model.transformers.set_transformer_attributes(k.split("__")[0], {k.split("__")[1]: val})
                    metr_exp.add_metr_val(MetricsVal(model_type=model_type,
                                                     model_name=model.name,
                                                     featsel_thresh=transformer_name,
                                                     expansion_type=metr_label,
                                                     metrics_type="best_params", metrics_name=k, val=val, usecase_name=usecase_name))
                else:
                    metr_exp.add_metr_val(MetricsVal(model_type=model_type,
                                                     model_name=model.name,
                                                     featsel_thresh=model.get_estimator().__class__.__name__,
                                                     expansion_type=metr_label,
                                                     metrics_type="best_params", metrics_name=k, val=val,
                                                     usecase_name=usecase_name))
                    setattr(model.get_estimator(), k.split("__")[1], val)

            # Train model
            model.fit(x_train, y_train)
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
                metr_val.set_metr_properties(model_type, model.name, metr_label, transformer_name, usecase_name)
            # Store metrics separately
            metr_exp.add_metr_vals(metr_vals)
            for type in ['Metrics', 'pvalues', 'FeatureSelect']:
                filename = os.path.join(result_dir_model, f"{type}_{experiment_name}.csv")
                metr_exp.store_metr_df(metr_exp.get_metr_df(type), filename)
            # Export results
            exp = ResultExport(results_root=result_dir_model, plot_enabled=True)
            exp.export_result(train_data, str(lookback_horizon), show_fig=False)
            exp.plot_enabled = False
            exp.export_model_properties(model, metr_label)
            exp.export_featsel_metrs(model, metr_label)
        # Store all metrics
        metr_exp.store_all_metrics(result_dir, index_col='expansion_type')
        print('Experiment finished')
