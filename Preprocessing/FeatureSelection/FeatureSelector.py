import ennemi
import numpy as np
from minepy.mine import MINE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectorMixin, f_regression, r_regression
from sklearn.linear_model import LinearRegression
from abc import abstractmethod
from ModelTraining.datamodels.datamodels.processing.feature_extension.StoreInterface import StoreInterface


class FeatureSelector(SelectorMixin, StoreInterface):
    """
        FeatureSelector - implements SelectorMixin interface, can be stored to pickle.
        Options:
            - omit_zero_samples: Omit zero samples from selection
    """
    coef_ = None
    nonzero = None
    omit_zero_samples=False
    nz_idx = None
    thresh = 0

    @staticmethod
    def from_name(name):
        dict_selectors = {'F-value': f_threshold, 'R-value': r_threshold, 'MIC-value': mine_mic_threshold,
                          'Ennemi-value': ennemi_threshold, 'forward_select': ForwardSelector,
                          'MIC-R-value': MIC_R_selector}
        selector_class = dict_selectors.get(name, identity)
        return selector_class

    @staticmethod
    def from_params(params):
        return FeatureSelector.from_name(params.sel_type)(thresh=params.threshold, omit_zero_samples=params.omit_zero_samples)

    def __init__(self, thresh, **kwargs):
        self.thresh = thresh
        self.omit_zero_samples = kwargs.pop('omit_zero_samples', False)

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit transformer - Overrides TransformerMixin method.
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        if X.ndim == 3:
            X = X.reshape((X.shape[0],X.shape[1] * X.shape[2]))
        self.nonzero = ~np.all(X == 0, axis=0)
        self.nz_idx = np.where(self.nonzero)
        if self.omit_zero_samples:
            coef = self._fit_transform(X[:, self.nonzero], y, **fit_params)
            self.coef_ = np.zeros(X.shape[-1])
            self.coef_[self.nz_idx] = coef
        else:
            self.coef_ = self._fit_transform(X, y, **fit_params)

        return X[:,self.get_support()]

    @abstractmethod
    def _fit_transform(self, X, y, **fit_params):
        """
        Fit transformer - Override this method!
        @param x: Input feature vector (n_samples, n_features) or (n_samples, lookback, n_features)
        @param y: Target feature vector (n_samples)
        """
        raise NotImplementedError()

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


class f_threshold(FeatureSelector):
    """
    F-Threshold:
    Threshold based on F-test of the Pearson correlation value.
    The F-test values are normalized between 0 and 1 for the smallest to highest value.
    """
    def _fit_transform(self, X, y=None, **fit_params):
        f_val = f_regression(X, y)[0]
        # Normalize f val
        return f_val / np.nanmax(f_val)


class r_threshold(FeatureSelector):
    """
    R-Threshold:
    Threshold based on absolute value of the Pearson correlation value.
    """
    def _fit_transform(self, X, y=None, **fit_params):
        if X.shape[-1] > 0:
            return np.abs(r_regression(X, y))
        return None


class identity(FeatureSelector):
    """
    Identity:
    All features are selected.
    """
    def _fit_transform(self, X, y, **fit_params):
        return np.ones(X.shape[-1])

    def _get_support_mask(self):
        return np.array([True] * self.coef_.shape[-1])


class mine_mic_threshold(FeatureSelector):
    """
    MIC-threshold
    Features are selected based on MIC.
    """
    def _fit_transform(self, X, y=None, **fit_params):
        n_features = X.shape[-1]
        coef = np.zeros(n_features)
        mine = MINE()
        for i in range(n_features):
            mine.compute_score(X[:, i], np.ravel(y))
            coef[i] = mine.mic()
        return coef


class ennemi_threshold(FeatureSelector):
    """
    ennemi-threshold
    Features are selected based on ennemi criterion.
    """
    def _fit_transform(self, X, y=None, **fit_params):
        return np.array([ennemi.estimate_corr(np.ravel(y), X[:,i], preprocess=True) for i in range(X.shape[-1])]).ravel()


class MIC_R_selector(FeatureSelector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mic_thresh = mine_mic_threshold(**kwargs)
        self.r_thresh = r_threshold(**kwargs)
        self.num_basic_features = kwargs.get('num_basic_features', 0)
        self.feature_names = np.array(kwargs.get('feature_names',[]))
        pass

    def fit_transform(self, X, y=None, **fit_params):
        self.mic_thresh.fit_transform(X, y, **fit_params)
        self.r_thresh.fit_transform(X, y, **fit_params)
        self.coef_ = np.array([self.mic_thresh.coef_, self.r_thresh.coef_]).T
        return X[self.get_support()]

    def _get_support_mask(self,indices=False):
        # Select features to expand based on MIC value
        basic_features_to_expand = self.mic_thresh.get_support()[:self.num_basic_features]
        feature_names_basic = self.feature_names[:self.num_basic_features]
        feature_names_to_expand = feature_names_basic[basic_features_to_expand]
        exp_mask = np.array([False] * len(self.feature_names))
        exp_mask[:self.num_basic_features][np.bitwise_not(basic_features_to_expand)] = True
        for name in feature_names_to_expand:
            #expanded_feature_names = [expanded_name for expanded_name in self.feature_names if name in expanded_name]
            #print(expanded_feature_names)
            support_mask_feature = np.array([name in self.feature_names[i] for i in range(len(self.feature_names))])
            exp_mask = np.bitwise_or(support_mask_feature, exp_mask)

        r_mask = self.r_thresh._get_support_mask()
        return np.bitwise_and(r_mask, exp_mask)


class ForwardSelector(FeatureSelector):
    def _fit_transform(self, X, y=None, **fit_params):
        f_model = LinearRegression()
        # efs = EFS(f_model,
        #           min_features=2,
        #           max_features=x.shape[1],
        #           scoring='neg_mean_squared_error',
        #           cv=3)
        self.sfs = SFS(f_model,
                  k_features='best',
                  forward=True,
                  n_jobs=-1)

        self.sfs.fit(X, y)
        self.num_features = X.shape[-1]
        self.coef_ = self.sfs.k_score_
        return self.coef_

    def _get_support_mask(self):
        f_indx = list(self.sfs.k_feature_idx_)
        return [True if i in f_indx else False for i in range(self.num_features)]