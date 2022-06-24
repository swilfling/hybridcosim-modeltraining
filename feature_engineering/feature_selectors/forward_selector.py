from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression

from .feature_selector import FeatureSelector


class ForwardSelector(FeatureSelector):
    def _fit(self, X, y=None, **fit_params):
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
        self.coef_ = self.sfs.k_score_

    def _get_support_mask(self):
        f_indx = list(self.sfs.k_feature_idx_)
        return [True if i in f_indx else False for i in range(self.n_features_in_)]