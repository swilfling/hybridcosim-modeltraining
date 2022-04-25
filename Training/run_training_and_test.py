import logging
import os
from typing import List

import pandas as pd
from sklearn.model_selection import GridSearchCV

from ModelTraining.TrainingUtilities import MetricsExport
import ModelTraining.TrainingUtilities.preprocessing
import ModelTraining.datamodels.datamodels.validation.white_test
from ModelTraining.TrainingUtilities import training_utils as train_utils
from ModelTraining.TrainingUtilities.MetricsExport import export_metrics as metr_exp
from ModelTraining.Training.predict import predict_with_history, predict_gt
from ModelTraining.Training.ModelCreation import create_model
from ModelTraining.Training.GridSearch import prepare_data_for_fit, create_pipeline
from ModelTraining.FeatureEngineering.FeatureSelection.FeatureSelector import FeatureSelector
import ModelTraining.FeatureEngineering.FeatureSelection.feature_selection as feat_select
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.FeatureEngineering.FeatureSelection.feature_selection_params import FeatureSelectionParams


def run_training_and_test(data, list_training_parameters: List[TrainingParams],
                          results_dir_path, do_predict=True, prediction_type="History", plot_enabled=True, **kwargs):
    models, results = [], []
    df_metrics_full = pd.DataFrame(index=[list_training_parameters[0].model_type])

    metrics_exp = MetricsExport(plot_enabled=plot_enabled, results_root=results_dir_path)
    # Get optional arguments
    metrics_names = kwargs.get('metrics_names', {'Metrics':['R2','CV-RMS', 'MAPE'], 'FeatureSelect':['selected_features', 'all_features'], 'pvalues':['pvalues_lm']})
    model_parameters = kwargs.get('model_parameters', None)
    expander_parameters = kwargs.get('expander_parameters',{})
    feature_select_params = kwargs.get('feature_select_params', [FeatureSelectionParams()])

    for training_params in list_training_parameters:
        target_features = training_params.target_features

        # Get data and reshape
        index, x, y, feature_names = ModelTraining.TrainingUtilities.preprocessing.extract_training_and_test_set(data, training_params)
        index_train, x_train, y_train, index_test, x_test, y_test = ModelTraining.TrainingUtilities.preprocessing.split_into_training_and_test_set(
            index, x, y, training_params.training_split)

        # Create model
        logging.info(f"Training model with input of shape: {x_train.shape} and targets of shape {y_train.shape}")
        model = create_model(training_params, expander_parameters=expander_parameters, feature_names=feature_names)

        # Select features + Grid Search
        selectors = [FeatureSelector.from_params(params) for params in feature_select_params]
        search = GridSearchCV(model.model, model_parameters,
                              scoring=['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error'], refit='r2',
                              verbose=4)
        pipeline = create_pipeline(model.expanders, selectors, search)
        pipeline.fit(*prepare_data_for_fit(model, x_train, y_train))
        # Set grid search params
        print(f"Best score for model {model.__class__.__name__} is: {search.best_score_}")
        print(f"Best parameters are {search.best_params_}")
        # Configure model
        model.model.set_params(**search.best_params_)
        feat_select.configure_feature_select(model.expanders, selectors)
        # Train model
        model.train(x_train, y_train)
        models.append(model)
        # Save Model
        train_utils.save_model_and_parameters(os.path.join(results_dir_path, f"Models/{training_params.model_name}/{training_params.model_type}_{training_params.expansion[0]}"), model, training_params)
        # Predict test data
        if do_predict:
            predict_function = predict_with_history if prediction_type == 'History' else predict_gt
            result_prediction = predict_function(model, index_test, x_test, y_test, training_params)

            for feature in training_params.target_features:
                y_pred = result_prediction[f"predicted_{feature}"].to_numpy()
                y_true = result_prediction[feature].to_numpy()[:y_pred.shape[0]]
                # Calculate Metrics
                metrics = metr_exp.calc_metrics(y_true, y_pred, x.shape[0], len(model.get_expanded_feature_names()),
                                         metrics_names=metrics_names['Metrics'])
                df_metrics_full = metr_exp.update_metr_df(df_metrics_full, metrics, prefix=f'{training_params.model_name}_{feature}_', suffix='_Metrics')
                # White test
                white_test_results = ModelTraining.datamodels.datamodels.validation.white_test.white_test(x_test, y_true - y_pred)
                df_metrics_full = metr_exp.update_metr_df(df_metrics_full, white_test_results, prefix=f'{training_params.model_name}_{feature}_', suffix='_pvalues')
            metrics_exp.export_results(model, target_features, result_prediction)

            test_prediction = result_prediction[[f"predicted_{feature}" for feature in target_features]].to_numpy()
            results.append(TrainingResults(train_index=index_train, train_target=y_train,
                                     test_index=index_test, test_target=y_test, test_prediction=test_prediction))

        # Export feature selection metrics
        metrics_exp.export_rvals(model.expanders, selectors, model.feature_names)
        for selector in selectors:
            df_metrics_full = metr_exp.update_metr_df(df_metrics_full, selector.get_metrics(),
                                                      prefix=f'{training_params.model_name}_{selector.__class__.__name__}_',
                                                      suffix='_FeatureSelect')

    return models, [results, df_metrics_full]