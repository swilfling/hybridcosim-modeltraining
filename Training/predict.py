import numpy as np
import pandas as pd
from collections import deque
from ..datamodels.datamodels.wrappers import AutoRecursive
from ..Utilities import feature_combination as fc


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
        dynamic_feature_vector = fc.create_dynamic_feature_vector(training_params.dynamic_output_features, queues, num_lookback_states)
        # Static features
        static_feats = {name: val for name, val in zip(training_params.static_input_features, list(row.flatten()))}
        static_feature_vector = fc.create_static_feature_vector(static_feats)
        # Combine both
        x = fc.combine_static_and_dynamic_features(static_feature_vector, dynamic_feature_vector)
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
    num_init_samples = training_params.lookback_horizon + 1
    # Get input features
    x_test_reshaped = x_test.reshape((x_test.shape[0], -1))
    if feature_names is not None:
        static_feat_indices = [feature in training_params.static_input_features for feature in feature_names]
        lagged_feat_names = [f'{feat}_{i}' for i in range(1, training_params.lookback_horizon + 1) for feat in training_params.dynamic_input_features]
        list_dyn_feat_names = training_params.dynamic_input_features + lagged_feat_names
        dynamic_feat_indices = [feature in list_dyn_feat_names for feature in feature_names]

        input_feats = x_test_reshaped[:, np.bitwise_or(static_feat_indices, dynamic_feat_indices)]
    else:
        input_feats = x_test_reshaped[:, :x_test.shape[-1] - y_true.shape[-1]]
    # Use first samples as start values, cut off first values from static feats
    start_values = y_true[:num_init_samples]
    input_feats = input_feats.reshape(input_feats.shape[0], 1, input_feats.shape[1])[num_init_samples:num_init_samples + prediction_length]

    ac = AutoRecursive(model)
    start_val_np = np.reshape(start_values, (training_params.lookback_horizon + 1, len(training_params.target_features)))
    result_ac = ac.predict(start_val_np, prediction_length, input_feats)
    result = np.concatenate((start_values, result_ac))
    return result
