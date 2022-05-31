import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats import outliers_influence as stats
from scipy.stats import boxcox
from scipy.stats import kstest
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller



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

########################## Sparsity ####################################################################################


def calc_sparsity_abs(data, threshold=0):
    return np.nansum(np.abs(data) > threshold) / data.shape[0]


############################ Density ###################################################################################

def calc_skew_kurtosis(data, omit_zero_samples=False):
    selected_metrics = ['skew','kurtosis']
    if omit_zero_samples:
        df = pd.DataFrame(columns=data.columns, index=selected_metrics)
        for feature in data.columns:
            data_feature = data[feature][data[feature] != 0]
            df[feature] = data_feature.agg(selected_metrics).values
        return df
    else:
        return data.agg(selected_metrics)

############################################ Transformations ###########################################################

def boxcox_transf(data:pd.DataFrame,omit_zero_samples=False, offset=0.000001):
    df = pd.DataFrame(columns=data.columns, index=data.index)
    for feat in data.columns:
        if omit_zero_samples:
            cur_feat = data[feat][data[feat] != 0]
        else:
            cur_feat = data[feat] + np.min(data[feat]) + offset
        df[feat] = boxcox(cur_feat)[0]
    return df

############################################# Tests ####################################################################

def norm_stat_tests(cur_data):
    df_tests = pd.DataFrame(index=cur_data.columns, columns=['Shapiro', 'KS', 'ADF'])
    for feat in cur_data:
        cur_feat = cur_data[feat]
        df_tests['Shapiro'][feat] = shapiro(cur_feat).pvalue
        df_tests['KS'][feat] = kstest(cur_feat, 'norm').pvalue
        df_tests['ADF'][feat] = adfuller(cur_feat, autolag="AIC")[1]
    return df_tests