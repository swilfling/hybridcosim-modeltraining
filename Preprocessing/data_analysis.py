import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats import outliers_influence as stats
from scipy.stats import boxcox
from scipy.stats import kstest
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller


########################### Variance Inflation Factor (VIF) ###########################################################


def calc_vif(X: np.ndarray):
    """
    Calculate Variance Inflation Factor for each feature in X
    @param X: array (n_samples, n_features)
    @return: array containing VIF values
    """
    return np.array([stats.variance_inflation_factor(X, i) for i in range(X.shape[-1])])


def calc_vif_df(df: pd.DataFrame, dropinf=False):
    """
    Calculate VIF for values, return pd Dataframe
    @param df: dataframe
    @param dropinf: drop infinite values from returned dataframe
    @return: dataframe containing VIF values
    """
    # Omit all-zero features for calculation
    select_mask = np.sum(np.abs(df.values), axis=0) > 0
    X_sel = df.values[:, select_mask]
    feature_names_sel = np.array(df.columns)[select_mask]
    df = pd.DataFrame(index=feature_names_sel,
                      data=calc_vif(X_sel), columns=['VIF'])
    if dropinf:
        df = df[df['VIF'] != np.inf]
    return df


####################### Correlation matrix ############################################################################


def corrmatrix(data: pd.DataFrame):
    """
    Calculate correlation matrix
    @param data: pd.Dataframe
    @return: Correlation matrix
    """
    label_encoder = LabelEncoder()
    data.iloc[:, 0] = label_encoder.fit_transform(data.iloc[:, 0]).astype('float64')
    corr = data.corr()
    corr = corr.dropna(axis=0, thresh=2)
    corr = corr.dropna(axis=1, thresh=1)
    return corr


def reshape_corrmatrix(corr: pd.DataFrame):
    """
    Reshape correlation matrix to dataframe with double index:
        e.g. xcoord  ycoord | correlation val
    @param corr: correlation matrix
    @return: df containing reshaped corrmatrix
    """
    n_cols = len(corr.columns)
    vals = corr.values.flatten()
    yvals = np.repeat(np.arange(0, n_cols), n_cols)
    xvals = np.mod(np.arange(0, n_cols * n_cols), n_cols)
    return pd.DataFrame(index=[xvals, yvals], data=vals, columns=['C'])

########################## Sparsity ####################################################################################


def calc_sparsity_abs(data: pd.DataFrame, threshold=0):
    """
    Calculate sparsity with threshold: absolute value of samples > significance threshold
    @param data: pandas dataframe
    @param threshold: significance threshold
    @return: fraction of samples with absolute value > threshold
    """
    return np.nansum(np.abs(data) > threshold) / data.shape[0]


############################ Density ###################################################################################

def calc_skew_kurtosis(data: pd.DataFrame, omit_zero_samples=False):
    """
    Calculate skew and kurtosis
    @param data: pandas dataframe
    @param omit_zero_samples: do not use zero samples in calculation
    @return: pd.Dataframe containing results
    """
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


def boxcox_transf(data: pd.DataFrame, omit_zero_samples=False, offset=0.000001):
    """
    Box-Cox Transformation
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html?highlight=boxcox#scipy.stats.boxcox
    @param data: pandas dataframe
    @param omit_zero_samples: do not transform zero samples
    @param offset: offset to prevent zero values if not omitting zero samples
    @return: transformed df
    """
    df = pd.DataFrame(columns=data.columns, index=data.index)
    for feat in data.columns:
        if omit_zero_samples:
            cur_feat = data[feat][data[feat] != 0]
        else:
            cur_feat = data[feat] + np.min(data[feat]) + offset
        df[feat] = boxcox(cur_feat)[0]
    return df

############################################# Tests ####################################################################


def norm_stat_tests(data: pd.DataFrame):
    """
    Statistical tests for normality: Shapiro-Wilk, Kolmogorov-Smirnov, Augmented Dickey Fuller
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html?highlight=shapiro#scipy.stats.shapiro
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html
    https://www.statsmodels.org/stable/examples/notebooks/generated/stationarity_detrending_adf_kpss.html?highlight=adf
    @param data: Dataframe to analyze
    @return: pd.Dataframe containing results
    """
    df_tests = pd.DataFrame(index=data.columns, columns=['Shapiro', 'KS', 'ADF'])
    for feat in data:
        cur_feat = data[feat]
        df_tests['Shapiro'][feat] = shapiro(cur_feat).pvalue
        print(shapiro(cur_feat))
        df_tests['KS'][feat] = kstest(cur_feat, 'norm').pvalue
        df_tests['ADF'][feat] = adfuller(cur_feat, autolag="AIC")[1]
    return df_tests


