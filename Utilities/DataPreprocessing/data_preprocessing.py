import numpy as np
import ModelTraining.datamodels.datamodels.validation.metrics as metrs
from . Filter import ButterworthFilter, ChebyshevFilter, Envelope_MA


def preprocess_data(data, features_to_smoothe, smoothe_data=False, keep_nans=False):
    if smoothe_data:
        filter = ButterworthFilter(order=2, T=10, keep_nans=keep_nans)
        filtered_vals = filter.fit_transform(data[features_to_smoothe].to_numpy())
        data[features_to_smoothe] = filtered_vals
    return data


def calculate_metrics(simulation_results, plotting_variables_lists):
    metrics_vals ={vars[1]: [metrs.all_metrics(np.array(results[vars[0]]), np.array(results[vars[1]])) for results in simulation_results] for vars in plotting_variables_lists}
    return metrics_vals


def demod_signals(df, labels, keep_nans=True, T_mod=20):
    '''
        Process modulated signals
    '''
    filt_labels = [f'{label}_filt' for label in labels]
    env_labels = [f'{label}_env' for label in labels]
    df[filt_labels] = ChebyshevFilter(T=T_mod * 10, keep_nans=keep_nans, remove_offset=True).fit_transform(df[labels])
    df[env_labels] = Envelope_MA(T=T_mod, keep_nans=keep_nans, remove_offset=True).fit_transform(df[labels])
    return df
