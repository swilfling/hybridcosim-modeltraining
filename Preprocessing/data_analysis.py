import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats import outliers_influence as stats


########################### Variable Inflation Factor (VIF) ###########################################################


def calc_vif(X):
    return np.array([stats.variance_inflation_factor(X, i) for i in range(X.shape[-1])])


def calc_vif_df(X, feature_names, dropinf=False):
    select_mask = np.sum(np.abs(X), axis=0) > 0
    print(select_mask)
    X_sel = X[:, select_mask]
    feature_names_sel = np.array(feature_names)[select_mask]
    df = pd.DataFrame(index=feature_names_sel,
                      data=calc_vif(X_sel), columns=['VIF'])
    if dropinf:
        df = df[df['VIF'] != np.inf]
    return df


####################### Correlation matrix ############################################################################


def corrmatrix(dataframe):
    label_encoder = LabelEncoder()
    dataframe.iloc[:, 0] = label_encoder.fit_transform(dataframe.iloc[:, 0]).astype('float64')
    corr = dataframe.corr()
    corr = corr.dropna(axis=0, thresh=2)
    corr = corr.dropna(axis=1, thresh=1)
    return corr


def reshape_corrmatrix(corr: pd.DataFrame):
    n_cols = len(corr.columns)
    vals = corr.values.flatten()
    yvals = np.repeat(np.arange(0, n_cols), n_cols)
    xvals = np.mod(np.arange(0, n_cols * n_cols), n_cols)
    return pd.DataFrame(index=[xvals, yvals], data=vals, columns=['C'])
