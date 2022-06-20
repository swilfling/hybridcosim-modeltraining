import numpy as np
import pandas as pd
from collections import deque
from ..datamodels.datamodels.wrappers import AutoRecursive
from ..Utilities import type_casting as tc


def predict_gt(model, df_index, x, y_true, training_params):
    """ Prediction function using ground truth values for testing ML model"""
    return model.predict(x)


def predict_with_history(model, index_test, x_test, y_true, training_params):
    """
    Predict function using history of dynamical features for testing ML
    """
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
    # loop over the testing dataset
    for index, row in enumerate(x_test):
        dynamic_feature_vector = tc.create_dynamic_feature_vector(training_params.dynamic_output_features, queues, num_lookback_states)
        # Static features
        static_feats = {name: val for name, val in zip(training_params.static_input_features, list(row.flatten()))}
        static_feature_vector = tc.create_static_feature_vector(static_feats)
        # Combine both
        x = tc.combine_static_and_dynamic_features(static_feature_vector, dynamic_feature_vector)
        # Do prediction
        predicted_vals = model.predict(x)[0]
        ## update the queue
        for feature, index in index_dyn_feats.items():
            queues[f'{feature}_queue'].append(predicted_vals[index])
        # update predictions
        for i, col in enumerate(predictions.columns):
            predictions[col][index] = predicted_vals[i]
    return predictions.to_numpy(dtype='float')


def predict_history_ar(model, index_test, x_test, y_true, training_params, prediction_length=None, feature_names=None):
    if training_params.dynamic_output_features == []:
        print("There are no dynamic target features - use the basic prediction.")
        return None
    num_static_feats = len(training_params.static_input_features)
    num_dynamic_input_feats = len(training_params.dynamic_input_features)
    num_init_samples = training_params.lookback_horizon + 1
    # Get input features
    if feature_names is not None:
        static_feat_indices = [i for i, feature in enumerate(feature_names) if
                               feature in training_params.static_input_features]
        dynamic_feat_indices = list(range(num_static_feats,num_static_feats+num_dynamic_input_feats))
        input_feats = x_test[:,:, static_feat_indices + dynamic_feat_indices]
    else:
        input_feats = x_test[:, :, :x_test.shape[-1] - y_true.shape[-1]]
    # Use first samples as start values, cut off first values from static feats
    start_values = y_true[:num_init_samples]
    input_feats = input_feats[num_init_samples:, :, :]
    prediction_length = len(index_test) - len(start_values) if prediction_length is None else prediction_length
    input_feats = input_feats[:prediction_length, :, :]
    result_ac = predict_autorecursive(model, training_params, start_values, input_feats, prediction_length)
    result = np.concatenate((start_values, result_ac))
    return result


def predict_autorecursive(model, training_params, start_values, input_features=None, prediction_length=None):
    """
    This is the auto-recursive prediction based on history.
     @param model: trained model - e.g. linear regression model
     @param start_values: start values for dynamic features - can be list if only 1 feature
     @param input_features: static feature values for prediction length
     @return: np array of results
    """
    ac = AutoRecursive(model)
    num_lookback_states = training_params.lookback_horizon + 1
    start_val_np = np.reshape(start_values, (num_lookback_states, len(training_params.target_features)))
    if input_features is not None:
        input_features = input_features.reshape(prediction_length, 1, input_features.shape[2] * input_features.shape[1])

    return ac.predict(start_val_np, prediction_length, input_features)
