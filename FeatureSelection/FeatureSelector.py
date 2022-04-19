import logging

import ennemi
import numpy as np
from minepy.mine import MINE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.feature_selection import SelectorMixin, f_regression, r_regression
from sklearn.linear_model import LinearRegression


class FeatureSelector(SelectorMixin):
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

    def __init__(self, thresh, **kwargs):
        self.thresh = thresh
        self.omit_zero_samples = kwargs.pop('omit_zero_samples', False)

    def fit_transform(self, X, y=None, **fit_params):
        if X.ndim == 3:
            X = X.reshape((X.shape[0],X.shape[1] * X.shape[2]))
        self.nonzero = ~np.all(X == 0, axis=0)
        self.nz_idx = np.where(self.nonzero)
        if self.omit_zero_samples:
            coef = self._fit_transform(X[:, self.nonzero], y, **fit_params)
            self.coef_ = np.zeros(X.shape[-1])
            self.coef_[self.nz_idx] = coef
        else:
            self.coef_ = self.fit_transform(X, y, **fit_params)
        return self

    # Override this method
    def _fit_transform(self, X, y, **fit_params):
        pass

    # Override this if necessary
    def _get_support_mask(self):
        return self.coef_ > self.thresh if not self.omit_zero_samples else (self.coef_ > self.thresh) & self.nonzero

    def get_num_features(self):
        return np.sum(self._get_support_mask() > 0)

    def get_metrics(self):
        return {'selected_features': np.sum(self.get_support()), 'all_features': len(self.get_support())}

    def print_metrics(self):
        logging.info(f'Selecting features: {np.sum(self.get_support())} of {len(self.get_support())}')


class f_threshold(FeatureSelector):
    def _fit_transform(self, X, y=None, **fit_params):
        f_val = f_regression(X, y)[0]
        # Normalize f val
        return f_val / np.nanmax(f_val)


class r_threshold(FeatureSelector):
    def _fit_transform(self, X, y=None, **fit_params):
        return np.abs(r_regression(X, y))


class identity(FeatureSelector):
    def _fit_transform(self, X, y, **fit_params):
        return np.ones(X.shape[-1])


class mine_mic_threshold(FeatureSelector):
    def _fit_transform(self, X, y=None, **fit_params):
        n_features = X.shape[-1]
        coef = np.zeros(n_features)
        mine = MINE()
        for i in range(n_features):
            mine.compute_score(X[:, i], np.ravel(y))
            coef[i] = mine.mic()
        return coef


class ennemi_threshold(FeatureSelector):
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
        print(self.coef_)
        return self

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