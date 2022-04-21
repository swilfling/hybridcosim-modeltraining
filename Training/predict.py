import numpy as np
import pandas as pd

from ModelTraining.TrainingUtilities import training_utils as train_utils
from ModelTraining.datamodels.datamodels.wrappers import AutoRecursive
from ModelTraining.Utilities import type_casting as tc
from collections import deque


def predict_gt(model, df_index, x, y_true, training_params):
    """ Prediction function using ground truth values for testing ML model"""
    y_pred = model.predict(x)
    labels = [f"predicted_{feature}" for feature in training_params.target_features]
    new_df = pd.DataFrame(y_true, columns=training_params.target_features, index=df_index)
    for index, label in enumerate(labels):
        new_df[label] = y_pred[:,index]
    new_df = new_df.dropna(axis=0)
    return new_df


def predict_with_history(model, index_test, x_test, y_true, training_params):
    """ Predict function using history of dynamical features for testing ML"""
    num_lookback_states = training_params.lookback_horizon + 1
    queues = {f'{feature_name}_queue': deque([0] * num_lookback_states, maxlen=num_lookback_states) for feature_name in training_params.dynamic_output_features}    # Fill queues with lookback values

    if len(training_params.dynamic_output_features) > 0:
        x_test = x_test[num_lookback_states:]
        index_test = index_test[num_lookback_states:]
        y_true = y_true[num_lookback_states:]
        # Initialize queues
        initial_values = y_true[:num_lookback_states]
        for n_feat, feature_name in enumerate(training_params.dynamic_output_features):
            for i in range(num_lookback_states):
                queues[f'{feature_name}_queue'].append(initial_values[i, n_feat])

    predictions = pd.DataFrame(index=index_test, columns=[f'predicted_{feature}' for feature in training_params.target_features])
    # Find dynamic output feature indices
    index_dyn_feats = {feature:np.where(training_params.target_features == feature) for feature in training_params.dynamic_output_features}

   # y = predict_autorecursive(model, training_params, feature_set,
   #                       static_features,
   #                       prediction_length=static_features.shape[0])

    # loop over the testing dataset
    for index, row in enumerate(x_test):
        dynamic_feature_vector = tc.create_dynamic_feature_vector(training_params.dynamic_output_features, queues, num_lookback_states)
        # Static features
        static_feature_vector = tc.create_static_feature_vector(training_params.static_input_features, train_utils.add_names_to_features(training_params.static_input_features, row))
        # Combine both
        x = tc.combine_static_and_dynamic_features(static_feature_vector, dynamic_feature_vector)
        # Do prediction
        predicted_vals = model.predict(x)[0]
        ## update the queue
        for feature, index in index_dyn_feats.items():
            queues[f'{feature}_queue'].append(predicted_vals[index])
        # update predictions
        for i, feature in enumerate(training_params.target_features):
            predictions[f'predicted_{feature}'][index] = float(predicted_vals[i])
    target_features = pd.DataFrame(y_true, columns=training_params.target_features, index=index_test)
    return target_features.join(predictions).astype('float')


def predict_autorecursive(model, training_params, feature_set,
                          static_features=None,
                          prediction_length=1000):
    """
    This is the auto-recursive prediction based on history.
     @param model: trained model - e.g. linear regression model
     @param start_values: start values for dynamic features - can be list if only 1 feature
     @param static_features: static feature values for prediction length
    """
    ac = AutoRecursive(model)
    num_lookback_states = training_params.lookback_horizon + 1
    queues = tc.create_ques(feature_set.dynamic_features, num_lookback_states)
    if training_params.dynamic_input_features:
        start_values = [queues[f"{feature}_queue"] for feature in training_params.dynamic_input_features]
        start_val_np = np.array(start_values)
        num_output_features = len(training_params.target_features)
        start_val_np = np.reshape(start_val_np, (num_lookback_states, num_output_features))

        if static_features is not None:
            num_static_features = len(training_params.static_input_features)
            static_features = static_features.reshape(prediction_length, 1, num_static_features)

        return ac.predict(start_val_np, prediction_length, static_features)
    else:
        raise ValueError("The auto-recursive prediction should not be called if there are no dynamic features.")