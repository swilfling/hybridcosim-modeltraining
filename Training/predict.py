import numpy as np
import pandas as pd

from ModelTraining.TrainingUtilities import training_utils as train_utils
from ModelTraining.datamodels.datamodels.wrappers import AutoRecursive
from ModelTraining.Utilities import type_casting as tc

""" Prediction function using ground truth values for testing ML model"""
def predict(model, x, y_true, training_params, df_index=None):
    y_pred = model.predict(x)
    labels = [f"predicted_{feature}" for feature in training_params.target_features]
    new_df = pd.DataFrame(y_true, columns=training_params.target_features, index=df_index)
    for index, label in enumerate(labels):
        new_df[label] = y_pred[:,index]
    new_df = new_df.dropna(axis=0)
    return new_df


""" Predict function using history of dynamical features for testing ML"""
def predict_with_history(model, data, training_params, feature_set):
    # import static feats data, shape should be (# samples, 1, # feats)
    static_features_names = feature_set.get_static_input_feature_names()
    dynamic_input_features_names = feature_set.get_dynamic_input_feature_names()
    target_feature_names = feature_set.get_output_feature_names()
    num_lookback_states = training_params.lookback_horizon + 1

    # set data for static features and the queues
    data_lookback = data[num_lookback_states:]
    target_features = data_lookback[target_feature_names]
    static_features = data_lookback[static_features_names].to_numpy()
    dynamic_input_features = data_lookback[dynamic_input_features_names].to_numpy()
    queues = tc.create_ques(feature_set.dynamic_features, num_lookback_states)
    # Fill queues with lookback values
    for i in range(num_lookback_states):
        for feature_name in dynamic_input_features_names:
            queues[f'{feature_name}_queue'].append(data[feature_name].to_numpy()[i])
    predictions = pd.DataFrame(index=data_lookback.index, columns=[f'predicted_{feature}' for feature in feature_set.get_output_feature_names()])

   # y = predict_autorecursive(model, training_params, feature_set,
   #                       static_features,
   #                       prediction_length=static_features.shape[0])

    # loop over the testing dataset
    for index, static_row, dynamic_row in zip(range(static_features.shape[0]), static_features, dynamic_input_features):
        # Dynamic features - Append current value to queue
        for i, feature_name in enumerate(dynamic_input_features_names):
            queues[f'{feature_name}_queue'].append(dynamic_row[i])
        dynamic_feature_vector = tc.create_dynamic_feature_vector(dynamic_input_features_names, queues, num_lookback_states)
        # Static features
        static_feature_vector = tc.create_static_feature_vector(static_features_names, train_utils.add_names_to_features(static_features_names, static_row))
        # Combine both
        x = tc.combine_static_and_dynamic_features(static_feature_vector, dynamic_feature_vector)
        # Do prediction
        predicted_vals = model.predict(x)[0]
        ## update the queue
        cnt_outp = 0
        for feature in feature_set.dynamic_features:
            if feature.feature_type == 'output':
                queues[f'{feature.name}_queue'].append(predicted_vals[cnt_outp])
                cnt_outp += 1
        for i, feature in enumerate(target_feature_names):
            predictions[f'predicted_{feature}'][index] = float(predicted_vals[i])

    return target_features.join(predictions).astype('float')


"""
 This is the auto-recursive prediction based on history. 
 @param model: trained model - e.g. linear regression model
 @param start_values: start values for dynamic features - can be list if only 1 feature
 @param static_features: static feature values for prediction length
"""
def predict_autorecursive(model, training_params, feature_set,
                          static_features=None,
                          prediction_length=1000):
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