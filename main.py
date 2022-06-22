import os
import logging
import numpy as np
import pathlib
from sklearn.model_selection import TimeSeriesSplit
from ModelTraining.feature_engineering.featureset import FeatureSet
from ModelTraining.dataimport.data_import import load_from_json
from ModelTraining.feature_engineering.parameters import TrainingParams
from ModelTraining.Utilities import TrainingResults
import ModelTraining.Training.TrainingUtilities.training_utils as train_utils
import ModelTraining.Utilities.Plotting.plot_data as plt_utils
from ModelTraining.Training.predict import predict_gt, predict_with_history
import ModelTraining.Preprocessing.data_preprocessing as dp_utils
from ModelTraining.dataimport import DataImport
import ModelTraining.datamodels.datamodels.validation.metrics as metrics
from ModelTraining.feature_engineering.feature_expanders import FeatureExpansion
from ModelTraining.datamodels.datamodels import Model
from ModelTraining.feature_engineering.expandedmodel import ExpandedModel, TransformerSet
from ModelTraining.datamodels.datamodels.processing import DataScaler


if __name__ == '__main__':
    hybridcosim_path = "../"
    config_path = os.path.join(hybridcosim_path,'ModelTraining', 'Configuration')
    usecase_config_path = os.path.join(hybridcosim_path,'ModelTraining', 'Configuration','UseCaseConfig')
    name = 'CPS-Data'
    dict_usecase = load_from_json(os.path.join(config_path,"UseCaseConfig", f"{name}.json"))
    data_import = DataImport.load(os.path.join(config_path, "DataImport", f"{dict_usecase['dataset_filename']}.json"))
    data = data_import.import_data(os.path.join(hybridcosim_path, dict_usecase['dataset_dir'], dict_usecase['dataset_filename']))
    feature_set = FeatureSet(os.path.join("./", dict_usecase['fmu_interface']))
    # Get training and target features
    target_features = feature_set.get_output_feature_names()
    input_features = feature_set.get_input_feature_names()

    # Added: Preprocessing - Smooth features
    smoothe_data = False
    data = dp_utils.preprocess_data(data, dict_usecase["to_smoothe"],dict_usecase['dataset_filename'], do_smoothe=smoothe_data)

    print("Starting Training")

    # Training parameters
    model_type = "RandomForestRegression"
    normalizer = "IdentityScaler"
    train_frac = 0.9
    prediction_horizon = 1
    lookback_horizon = 4

    training_results_path = os.path.join(hybridcosim_path, "ModelTraining/results/")

    plot_dir_name = "results/Plots"

    prediction_options = {
        "ground_truth": predict_gt,
        "history": predict_with_history
    }

    predict_type = "ground_truth"

    list_training_parameters = []
    models = []
    results = []

    expansion = ["PolynomialExpansion"]

    static_feature_names = feature_set.get_static_feature_names()
    static_feature_data = data[static_feature_names]

    cross_validation = False

    for feature in target_features:
        static_features = feature_set.get_static_feature_names(feature)
        dynamic_features = feature_set.get_dynamic_feature_names(feature)
        training_parameters = TrainingParams(model_type=model_type,
                                             model_name=feature,
                                             lookback_horizon=lookback_horizon,
                                             target_features=[feature],
                                             prediction_horizon=prediction_horizon,
                                             static_input_features=static_features,
                                             dynamic_input_features=dynamic_features,
                                             training_split=train_frac,
                                             normalizer=normalizer,
                                             expansion=expansion)

        # Get data and reshape
        index, x, y, _ = train_utils.extract_training_and_test_set(data, training_parameters)
        ## Training process
        model_basic = Model.from_name(training_parameters.model_type,
                                x_scaler_class=DataScaler.cls_from_name(training_parameters.normalizer),
                                name=training_parameters.str_target_feats(), parameters={})
        model = ExpandedModel(model=model_basic, transformers=TransformerSet(FeatureExpansion.from_names(expansion)),
                              feature_names=training_parameters.static_input_features + training_parameters.dynamic_input_features)
        rmse_best = None
        train_ind_best = []
        test_ind_best = []
        if cross_validation:
            ts_kf = TimeSeriesSplit(n_splits=4)
            for train_ind, test_ind in ts_kf.split(x, y):
                x_train_k, x_test_k = x[train_ind, :], x[test_ind, :]
                y_train_k, y_test_k = y[train_ind], y[test_ind]
                model.train(x_train_k, y_train_k)
                result_k = TrainingResults(train_index=train_ind, train_target=y_train_k,
                                           test_index=test_ind, test_target=y_test_k)
                result_prediction_k = predict_with_history(model, test_ind, x_test_k, y_test_k, training_parameters)
                result_k.test_prediction = np.expand_dims(result_prediction_k[f"predicted_{feature}"], axis=-1)
                measures = metrics.all_metrics(y_true=result_k.test_target, y_pred=result_k.test_prediction)
                if rmse_best is None:
                    rmse_best = measures["CV-RMS"]
                    train_ind_best = train_ind
                    test_ind_best = test_ind
                else:
                    if rmse_best > measures["CV-RMS"]:
                        rmse_best = measures["CV-RMS"]
                        train_ind_best = train_ind
                        test_ind_best = test_ind

            index_train, x_train, y_train, index_test, x_test, y_test = index[train_ind_best], x[train_ind_best, :], y[
                train_ind_best], \
                                                                        index[test_ind_best], x[test_ind_best, :], y[
                                                                            test_ind_best]
        else:
            index_train, x_train, y_train, index_test, x_test, y_test = train_utils.split_into_training_and_test_set(
                index, x, y, training_parameters.training_split)

        logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")

        model.train(x_train, y_train)
        models.append(model)

        result_prediction = predict_with_history(model, index_test, x_test, y_test, training_parameters)
        result = TrainingResults(train_index=index_train, train_target=y_train,
                                 test_index=index_test, test_target=y_test,
                                 test_prediction=result_prediction, target_feat_names=target_features)
        title = f"{training_parameters.model_type} - {training_parameters.model_name}"

        plot_dir = pathlib.Path(os.path.join(training_results_path, plot_dir_name))
        os.makedirs(os.path.join(training_results_path, plot_dir_name), exist_ok=True)
        #print(result_prediction.columns)
        plt_utils.plot_data(result.test_result_df(), plot_dir, filename=title, fig_title=title)
        results.append(result)
        # Calculate metrics
        metrs = []
        # Check lengths for metrics
        for result in results:
            for feat in result.target_feat_names:
                y_true = result.test_target_vals(feat)
                y_pred = result.test_pred_vals(feat)
                metrs.append(metrics.all_metrics(y_true=y_true, y_pred=y_pred))
                plt_utils.scatterplot(y_pred.flatten(), y_true.flatten(), './results/', f'Scatterplot_{feat}')
        print("Metrics:")
        print(metrs)
        # Save Model
        train_utils.save_model_and_params(model, training_parameters, training_results_path)



    print('Experiment finished')