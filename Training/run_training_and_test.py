import logging
import os
from typing import List

import pandas as pd

import ModelTraining.TrainingUtilities.preprocessing
import ModelTraining.datamodels.datamodels.validation.white_test
from ModelTraining.TrainingUtilities import export_metrics as metr_exp, training_utils as train_utils
from ModelTraining.Training.predict import predict_with_history, predict
from ModelTraining.Training.ModelCreation import create_model
from ModelTraining.Training.GridSearch import best_estimator
from ModelTraining.FeatureSelection.feature_selection import create_selector_pipeline
from ModelTraining.FeatureSelection.FeatureSelector import FeatureSelector
from ModelTraining.Utilities.Parameters import TrainingParams, TrainingResults
from ModelTraining.FeatureSelection.feature_selection_params import FeatureSelectionParams
from ModelTraining.Utilities.feature_set import FeatureSet


def run_training_and_test(data, list_training_parameters: List[TrainingParams],
                          results_dir_path, do_predict=True, prediction_type="History", plot_enabled=True,
                          metrics_names={'Metrics':['R2','CV-RMS', 'MAPE'], 'FeatureSelect':['selected_features', 'all_features'], 'pvalues':['pvalues_lm']},
                          **kwargs):
    models, results = [], []
    df_metrics_full = pd.DataFrame(index=[list_training_parameters[0].model_type])

    metrics_exp = metr_exp.MetricsExport(plot_enabled=plot_enabled, results_root=results_dir_path)
    # Get optional arguments
    model_parameters = kwargs.get('model_parameters', None)
    expander_parameters = kwargs.get('expander_parameters',{})
    feature_set = kwargs.get('feature_set', FeatureSet())
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

        # Select features
        selectors = [FeatureSelector.from_params(params) for params in feature_select_params]
        pipeline = create_selector_pipeline(model.expanders, selectors)
        pipeline.fit_transform(x_train, y_train)
        # Set feature select for model
        for expander, selector in zip(model.expanders, selectors):
            expander.set_feature_select(selector.get_support())
            selector.print_metrics()
        # Export feature selection metrics
        metrics_exp.export_rvals(model.expanders, selectors, model.feature_names)

        # Grid search
        best_params = best_estimator(model, x_train, y_train, parameters=model_parameters)
        model.model.set_params(**best_params)
        # Train model
        model.train(x_train, y_train)
        models.append(model)

        # Save Model
        train_utils.save_model_and_parameters(os.path.join(results_dir_path, f"Models/{training_params.model_name}/{training_params.model_type}_{training_params.expansion[0]}"), model, training_params)
        # Predict test data
        if do_predict:
            if prediction_type == "History":
                test_data = data[training_params.static_input_features + training_params.dynamic_input_features + target_features].loc[index_test]
                result_prediction = predict_with_history(model, test_data, training_params, feature_set)
            else:
                result_prediction = predict(model, x_test, y_test, training_params, index_test)

            for feature in training_params.target_features:
                y_true = result_prediction[feature].to_numpy()
                y_pred = result_prediction[f"predicted_{feature}"].to_numpy()
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

        for selector in selectors:
            df_metrics_full = metr_exp.update_metr_df(df_metrics_full, selector.get_metrics(),
                                                      prefix=f'{training_params.model_name}_{selector.__class__.__name__}_',
                                                      suffix='_FeatureSelect')

    return models, [results, df_metrics_full]