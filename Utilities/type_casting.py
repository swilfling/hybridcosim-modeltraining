import numpy as np
from collections import deque


def cast(variable, totype: str):
    """
    Cast a variable to a type, where type is a string variable
    Can cast to base types: float, int, bool and string
    Parameters
    ----------
    variable The variable to cast
    totype The type to cast into, as a string:
        Real   -> float
        Int    -> int
        Bool   -> bool
        String -> str

    Returns totype(variable) or original type if type not found
    -------

    """
    castingtype = totype.lower()
    if castingtype == "real":
        return float(variable)
    if castingtype == "int":
        return int(variable)
    if castingtype == "bool":
        return bool(variable)
    if castingtype == "string":
        return str(variable)
    return variable


def create_ques(features, num_lookback_states=4):
    if features is not None:
        return {que_name(feature.name): deque([cast(feature.init, feature.datatype)] * num_lookback_states, maxlen=num_lookback_states) for feature in features}


def init_queue(queue, start_val):
    queue[:] = start_val


def init_queues(queues, start_vals):
    for name, val in start_vals.items():
        queues[que_name(name)][:] = val
    return queues


def que_name(feature_name):
    return f"{feature_name}_queue"

"""
DYNAMIC FEATURES
shape: 1 sample, lookback horizon + 1, number of features
turns into 1, 1, (lookback horizon + 1 * number of features)
"""
def create_dynamic_feature_vector(dynamic_features_names, dict_queues, queue_len=5):
    dynamic_features = np.array([dict_queues[que_name(name)] for name in dynamic_features_names])
    return dynamic_features.reshape((1, 1, len(dynamic_features_names) * queue_len))


"""
STATIC FEATURES
shape: 1 sample, 1, number of features
"""
def create_static_feature_vector(static_features_names, dict_static_values):
    if dict_static_values is not None:
        static_features = np.reshape(np.array([dict_static_values[name] for name in static_features_names]), (1, len(static_features_names)))
        return static_features.reshape((1, 1, static_features.shape[1]))


"""
Combine static and dynamic features
"""
def combine_static_and_dynamic_features(static_features: np.ndarray=None, dynamic_features: np.ndarray=None):
    if static_features is None:
        return dynamic_features
    if dynamic_features is None:
        return static_features
    """
       RETURNS shape: number of samples, 1, (lookback horizon + 1) * number of features)
    """
    return np.dstack([static_features, dynamic_features])


