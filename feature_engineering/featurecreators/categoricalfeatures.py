import numpy as np
import pandas as pd

from .featurecreator import FeatureCreator


class CategoricalFeatures(FeatureCreator):
    """
    Categorical Encoding - One-hot encoding
    Currently supported: weekday and hour
    """
    _onehot_vals = {"weekday": ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'],
                   "hour": [f"hour_{i}" for i in range(24)],
                   "month": [f"month_{i}" for i in range(12)]}

    def __init__(self, selected_feats=['weekday', 'hour'], **kwargs):
        super().__init__(selected_feats=selected_feats, **kwargs)

    def transform(self, X):
        """
        Add cyclic features
        @param X: input data
        @return: transformed data
        """
        X_t = X.copy()
        # Add labels to data
        for label in self.selected_feats:
            labels = self._onehot_vals[label]
            time = getattr(X.index, label)
            for weekday, day in zip(labels, range(len(labels))):
                X_t[weekday] = pd.to_numeric(time == day)
        return X_t

    def get_additional_feat_names(self):
        """
        Get names of cyclic features
        @return: list of feature names
        """
        list_feat_names = []
        for label in self.selected_feats:
            list_feat_names += self._onehot_vals[label]
        return list_feat_names


class CategoricalFeaturesDivider(CategoricalFeatures):
    """
    Categorical features with timestep divider
    e.g. feature: hour, divider: 2 will return
    hour_0, hour_2, hour_4 ,.....
    """
    division_factors: dict

    def __init__(self, selected_feats=['weekday', 'hour'], division_factors={'weekday':1, 'hour':2}, **kwargs):
        super().__init__(selected_feats, **kwargs)
        self.division_factors = division_factors

    def transform(self, X):
        """
        Add cyclic features
        @param X: input data
        @return: transformed data
        """
        X_t = X.copy()
        # Add labels to data
        for label in self.selected_feats:
            time = getattr(X.index, label)
            range_time = np.arange(0,len(self._onehot_vals[label]), self.division_factors[label])
            for val in range_time:
                X_t[self._onehot_vals[label][val]] = pd.to_numeric(time < val + self.division_factors[label]) * pd.to_numeric(time >= val)
        return X_t

    def get_additional_feat_names(self):
        """
        Get names of cyclic features
        @return: list of feature names
        """
        list_feat_names = []
        for label in self.selected_feats:
            list_feat_names += [self._onehot_vals[label][i] for i in np.arange(0,len(self._onehot_vals[label]), self.division_factors[label])]
        return list_feat_names