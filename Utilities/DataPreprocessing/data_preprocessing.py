import numpy as np
import ModelTraining.datamodels.datamodels.validation.metrics as metrs
from . LowpassFilter import ButterworthFilter


def preprocess_data(data, features_to_smoothe, smoothe_data=False, keep_nans=False):
    if smoothe_data:
        filter = ButterworthFilter(order=2, T=10, keep_nans=keep_nans)
        filtered_vals = filter.fit_transform(data[features_to_smoothe].to_numpy())
        data[features_to_smoothe] = filtered_vals
    return data


def calculate_metrics(simulation_results, plotting_variables_lists):
    metrics_vals ={vars[1]: [metrs.all_metrics(np.array(results[vars[0]]), np.array(results[vars[1]])) for results in simulation_results] for vars in plotting_variables_lists}
    return metrics_vals

