import numpy as np
import pandas as pd
from statsmodels.stats import outliers_influence as stats


########################### Variable Inflation Factor (VIF) ###########################################################


def calc_vif(X):
    return np.array([stats.variance_inflation_factor(X, i) for i in range(X.shape[-1])])


def calc_vif_df(X, feature_names, dropinf=False):
    df = pd.DataFrame(index=feature_names,
                 data=calc_vif(X), columns=['VIF'])
    if dropinf:
        df = df[df['VIF'] != np.inf]
    return df