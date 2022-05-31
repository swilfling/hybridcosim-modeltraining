import ennemi
import numpy as np
from minepy.mine import MINE
from sklearn.feature_selection import f_regression, r_regression
from .. FeatureSelection import FeatureSelector


class f_threshold(FeatureSelector):
    """
    F-Threshold:
    Threshold based on F-test of the Pearson correlation value.
    The F-test values are normalized between 0 and 1 for the smallest to highest value.
    """
    def _fit(self, X, y=None, **fit_params):
        f_val = f_regression(X, y)[0]
        # Normalize f val
        return f_val / np.nanmax(f_val)


class r_threshold(FeatureSelector):
    """
    R-Threshold:
    Threshold based on absolute value of the Pearson correlation value.
    """
    def _fit(self, X, y=None, **fit_params):
        if X.shape[-1] > 0:
            return np.abs(r_regression(X, y))
        return None


class mine_mic_threshold(FeatureSelector):
    """
    MIC-threshold
    Features are selected based on MIC.
    """
    def _fit(self, X, y=None, **fit_params):
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
    def _fit(self, X, y=None, **fit_params):
        vals = [ennemi.estimate_corr(np.ravel(y), X[:,i], preprocess=True) for i in range(X.shape[-1])]
        return np.array(vals).ravel()


class MIC_R_selector(FeatureSelector):
    """
    This is work in progress.
    """
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

