import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin

from ModelTraining.Preprocessing.FeatureSelection import feature_selectors
from ModelTraining.Preprocessing.FeatureSelection import FeatureSelectionParams
from ModelTraining.datamodels.datamodels.processing.feature_extension import StoreInterface


class FeatureSelector(SelectorMixin, BaseEstimator, StoreInterface):
    """
        FeatureSelector - implements SelectorMixin interface, can be stored to pickle.
        Basic implementation: threshold
        Options:
            - omit_zero_samples: Omit zero samples from selection
    """
    coef_ = None
    nonzero = None
    omit_zero_samples=False
    nz_idx = None
    thresh = 0
    n_features_in_ = 0

    @staticmethod
    def from_name(name):
        """
        Get selector by name
        @param name: selector name
        @return FeatureSelector object
        """
        dict_selectors = {'F-value': 'FThreshold', 'R-value': 'RThreshold', 'MIC-value': 'MICThreshold',
                          'Ennemi-value': 'EnnemiThreshold', 'forward_select': 'ForwardSelector',
                          'MIC-R-value': 'MIC_R_selector', 'Name': 'SelectorByName'}
        selector_class = getattr(feature_selectors, dict_selectors.get(name, 'IdentitySelector'))
        return selector_class

    @staticmethod
    def from_params(params: FeatureSelectionParams):
        """
        Get selector by name
        @param name: selector name
        @return FeatureSelector object
        """
        return FeatureSelector.from_name(params.sel_type)(thresh=params.threshold, omit_zero_samples=params.omit_zero_samples)

    @staticmethod
    def configure_feature_select(expanders, selectors):
        """
        Configure feature select for multiple expanders and selectors
        """
        for expander, selector in zip(expanders, selectors):
            expander.set_feature_select(selector.get_support())
            selector.print_metrics()

    def __init__(self, thresh=0, omit_zero_samples=False, **kwargs):
        self._set_attrs(thresh=thresh, omit_zero_samples=omit_zero_samples)

    def fit(self, X, y=None, **fit_params):
        """
        Fit transformer - Overrides TransformerMixin method.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        X = self.reshape_data(X)
        self.nonzero = ~np.all(X == 0, axis=0)
        self.nz_idx = np.where(self.nonzero)
        self.n_features_in_ = X.shape[-1]
        if self.omit_zero_samples:
            coef = self._fit(X[:, self.nonzero], y, **fit_params)
            self.coef_ = np.zeros(X.shape[-1])
            self.coef_[self.nz_idx] = coef
        else:
            self.coef_ = self._fit(X, y, **fit_params)
        return self

    def transform(self, X):
        """
        Transform samples.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @return: Output feature vector (n_samples, n_features) or (n_samples, n_selected_features * lookback)
        """
        return super(FeatureSelector, self).transform(self.reshape_data(X))

    def _fit(self, X, y, **fit_params):
        """
        Fit transformer - Override this method!
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        @return coefficients
        """
        return None

    def _get_support_mask(self):
        """
        Get boolean mask of selected features - override if necessary.
        """
        return self.coef_ > self.thresh if not self.omit_zero_samples else (self.coef_ > self.thresh) & self.nonzero

    def get_num_selected_features(self):
        """
        Get number of selected features - override if necessary.
        """
        return np.sum(self._get_support_mask())

    def get_num_features(self):
        """
        Get total number of features - override if necessary.
        """
        return self.coef_.shape[0]

    def get_metrics(self):
        """
        return selected and total features
        """
        return {'selected_features': self.get_num_selected_features(), 'all_features': self.get_num_features()}

    def print_metrics(self):
        """
        Print number of selected and total features
        """
        print(f'Selecting features: {self.get_num_selected_features()} of {self.get_num_features()}')

    def get_coef(self):
        """
        get coefficients for selected features
        """
        return self.coef_[self.get_support()]

    def reshape_data(self, X):
        """
        Reshape data if 3-dimensional.
        """
        if X.ndim == 3:
            X = X.reshape((X.shape[0], X.shape[1] * X.shape[2]))
        return X