import numpy as np


def init_queues(queues: dict, start_vals: dict):
    """
    Initialize queues.
    @param queues: Dict of queues
    @param start_vals: dict of start vals
    @return: initialized queues
    """
    for name, val in start_vals.items():
        queues[que_name(name)][:] = val
    return queues


def que_name(feature_name):
    """ Add queue suffix to feature name """
    return f"{feature_name}_queue"


def create_dynamic_feature_vector(dynamic_features_names, dict_queues, queue_len=5):
    """
    DYNAMIC FEATURES
    shape: 1 sample, lookback horizon + 1, number of features
    turns into 1, 1, (lookback horizon + 1 * number of features)
    """
    dynamic_features = np.array([dict_queues[que_name(name)] for name in dynamic_features_names])
    return dynamic_features.reshape((1, 1, len(dynamic_features_names) * queue_len))


def create_static_feature_vector(dict_static_feats):
    """
    STATIC FEATURES
    shape: 1 sample, 1, number of features
    """
    if dict_static_feats is not None:
        static_features = np.reshape(np.array(list(dict_static_feats.values())), (1, len(dict_static_feats)))
        return static_features.reshape((1, 1, static_features.shape[1]))


def combine_static_and_dynamic_features(static_features: np.ndarray=None, dynamic_features: np.ndarray=None):
    """
    Combine static and dynamic features
    """
    if static_features is None:
        return dynamic_features
    if dynamic_features is None:
        return static_features
    """
       RETURNS shape: number of samples, 1, (lookback horizon + 1) * number of features)
    """
    return np.dstack([static_features, dynamic_features])


