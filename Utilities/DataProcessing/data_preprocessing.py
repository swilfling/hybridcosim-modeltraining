import numpy as np
import ModelTraining.datamodels.datamodels.validation.metrics as metrs
import ModelTraining.Utilities.DataProcessing.signal_processing_utils as sigutils


def smoothe_features(df, features, keep_nans=False):
    df = df.copy()
    for feature in features:
        signal, mask = sigutils.remove_nans(df[feature])
        signal = sigutils.filter_smoothe_signal(signal,T=10)
        if keep_nans:
            sigutils.apply_nans(signal, mask)
        df[feature] = signal
    return df


def preprocess_data(data, feature_set, smoothe_data=False):
    keep_nans=False
    return smoothe_features(data, feature_set.get_output_feature_names() + feature_set.get_input_feature_names(), keep_nans) if smoothe_data else data


def calculate_metrics(simulation_results, plotting_variables_lists):
    metrics_vals ={vars[1]: [metrs.all_metrics(np.array(results[vars[0]]), np.array(results[vars[1]])) for results in simulation_results] for vars in plotting_variables_lists}
    return metrics_vals

