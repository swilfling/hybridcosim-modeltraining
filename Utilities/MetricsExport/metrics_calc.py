import os
import pandas as pd

from typing import List
from .metrics_vals import MetricsVal, MetrValsSet
from ...feature_engineering.expandedmodel import ExpandedModel
from ...datamodels.datamodels.validation import metrics
from ...datamodels.datamodels.validation.whitetest import white_test
from ..trainingdata import TrainingData
from ...feature_engineering.featureengineering.featureselectors import FeatureSelector


class MetricsCalc:
    metr_names = {'Metrics': ['rsquared', 'cvrmse', 'mape'],
                  'FeatureSelect': ['selected_features', 'all_features'],
                  'pvalues': ['pvalue_lm']}
    metr_vals: MetrValsSet = MetrValsSet()
    df_float_fmt = "%3f"

    def __init__(self, metr_names=None, df_float_fmt="%.3f"):
        self.df_float_fmt = df_float_fmt
        if metr_names is not None:
            self.metr_names = metr_names

    ################################# Metrics calculation #############################################################

    def calc_perf_metrics(self, result: TrainingData, n_predictors=0):
        list_metrs = []
        for feat in result.target_feat_names:
            metrs = self.perf_metr_dict(y_true=result.test_target_vals(feat),
                                        y_pred=result.test_pred_vals(feat),
                                        n_predictors=n_predictors,
                                        perf_metr_names=self.metr_names['Metrics'])
            list_metrs += [MetricsVal(metrics_type='Metrics', metrics_name=k, val=v, target_feat=feat) for k, v in metrs.items()]
        return list_metrs

    def analyze_featsel(self, selectors):
        """
        Analyze feature selection
        @param selectors: List of selectors
        @return: pd.DataFrame containing results
        """
        list_metrs = []
        for i, selector in enumerate(selectors):
            sel_metrs = selector.get_metrics()
            selector_name = f'{selector.__class__.__name__}_{i}'
            for k, v in sel_metrs.items():
                list_metrs.append(MetricsVal(metrics_type='FeatureSelect', metrics_name=f'{selector_name}_{k}', val=v))
        return list_metrs

    def white_test(self, result: TrainingData):
        """
        White test for all target features in the training result.
        @param result: TrainingResult structure
        """
        list_metr_vals = []
        for feat in result.target_feat_names:
            residual = result.test_target_vals(feat) - result.test_pred_vals(feat)
            residual = residual.to_numpy() if isinstance(residual, (pd.Series, pd.DataFrame)) else residual
            white_pvals = white_test(result.test_input, residual)
            for k, v in white_pvals.items():
                if k in self.metr_names['pvalues']:
                    list_metr_vals.append(MetricsVal(metrics_type='pvalues', metrics_name=k, val=v, target_feat=feat))
        return list_metr_vals

    def calc_all_metrics(self, result: TrainingData, selectors:List[FeatureSelector], num_predictors=0):
        """
        Calculate metrics vals for all metrics
        @param result: TrainingResult structure
        @param selectors: List of feature selectors
        @param num_predictors: needed for performance metric calc
        @return: list of metrics vals
        """
        return self.calc_perf_metrics(result,num_predictors) + self.analyze_featsel(selectors) + self.white_test(result)

    ####################################  Dataframe ###################################################################

    def white_test_df(self, result: TrainingData, df_index=None):
        """
        White test for all target features in the training result.
        @param result: TrainingResult structure
        @param selected_metrics: Selected metrics for white test. Supported: 'pvalue_lm', 'pvalue_f'
        @param df_index: Optional: index for dataframe
        """
        df_white = pd.DataFrame()
        for feat in result.target_feat_names:
            white_pvals = white_test(result.test_input, result.test_target_vals(feat) - result.test_pred_vals(feat))
            df_white_feat = pd.DataFrame(data=white_pvals.values(), index=white_pvals.keys()).transpose()[self.metr_names['pvalues']].add_suffix(f'_{feat}')
            df_white = df_white_feat if df_white.empty else df_white.join(df_white_feat)
        if df_index is not None:
            df_white.index = df_index
        return df_white

    def calc_perf_metrics_df(self, result: TrainingData, n_predictors=0, df_index=["Model1"]):
        """
        Calculate metrics
        @param result: TrainingResults structure
        @param n_predictors: number of predictors - only used for R_adj metric
        @param df_index: Optional: index for dataframe
        @return pd.Dataframe containing metrics
        """
        metrs = {}
        for feat in result.target_feat_names:
            metrs.update(self.perf_metr_dict(y_true=result.test_target_vals(feat),
                                             y_pred=result.test_pred_vals(feat),
                                             n_predictors=n_predictors,
                                             perf_metr_names=self.metr_names['Metrics'],
                                             key_suffix=f'_{feat}'))
        return pd.DataFrame(metrs, index=df_index)

    @staticmethod
    def perf_metr_dict(y_true, y_pred, n_predictors=0, perf_metr_names=['rsquared'], key_suffix=""):
        """
        Calculate performance metrics
        @param y_true: True vals
        @param y_pred: predicted vals
        @param n_predictors: number of predictors
        @param perf_metr_names: names of metrics
        @param key_suffix: suffix for dict key
        @return: dict containing result
        """
        metrs = {f'{name}{key_suffix}': float(getattr(metrics, name)(y_true=y_true, y_pred=y_pred)) for name in perf_metr_names}
        if 'RA' in perf_metr_names:
            metrs.update({f"RA{key_suffix}": metrics.adjusted_rsquared(y_true, y_pred, n_predictors)})
        return metrs

    def analyze_featsel_df(self, model: ExpandedModel, df_index=None):
        """
        Analyze feature selection
        @param model: Model to analyze - expanders are needed to get feature names
        @param selectors: List of selectors
        @param df_index: Optional: index for dataframe
        @return: pd.DataFrame containing results
        """
        model_name = f'{model.name}_{model.transformers.type_transf_full()}'
        df_featsel_full = pd.DataFrame()
        selectors = model.transformers.get_transformers_of_type(FeatureSelector)
        for i, selector in enumerate(selectors):
            sel_metrs = selector.get_metrics()
            selector_name = f'{selector.__class__.__name__}_{i}'
            df_featsel = pd.DataFrame(data=sel_metrs.values(), index=sel_metrs.keys()).transpose()
            df_featsel = df_featsel[self.metr_names['FeatureSelect']].add_suffix(f'_{selector_name}_{model_name}_FeatureSelect')
            df_featsel_full = df_featsel if df_featsel_full.empty else df_featsel_full.join(df_featsel)
        df_featsel_full.index = [model.__class__.__name__]
        if df_index is not None:
            df_featsel_full.index = df_index
        return df_featsel_full

    ############################################# Store metrics ########################################################

    def add_metr_val(self, metr_val: MetricsVal):
        """
        Add metrics to internal metrics set
        @param metr_val: MetricsVal struct
        """
        self.metr_vals.add_metr_val(metr_val)

    def add_metr_vals(self, metrs: List[MetricsVal]):
        """
        Add metrics to internal metrics set
        @param metrs: list of metrics
        """
        self.metr_vals.add_metr_vals(metrs)

    def get_metr_df(self, metrics_type="", index_col='model_type'):
        """
        Get dataframe containing currently stored metrics
        @return: pd Dataframe
        """
        return self.metr_vals.create_df_metrics(metrics_type=metrics_type, index_col=index_col)

    ############################ Store functions #######################################################################

    def store_all_metrics(self, results_path="./", timestamp="", index_col='model_type', index_label='Model'):
        """
        Store internal metrics to files - get subsets of dataframes to store separately
        @param results_path: result dir - optional argument
        @param timestamp: timestamp of experiment
        """
        df_metrs = self.metr_vals.create_df_metrics("",index_col=index_col)
        df_metrs.to_csv(os.path.join(results_path, f'AllMetrics_full_{timestamp}.csv'),index_label='Model', float_format=self.df_float_fmt)

        # Get subsets
        df_metrics = self.metr_vals.create_df_metrics('Metrics', index_col=index_col)
        df_featsel = self.metr_vals.create_df_metrics('FeatureSelect', index_col=index_col)
        df_whitetest = self.metr_vals.create_df_metrics('pvalues', index_col=index_col)

        # Store results
        for df, filename in zip([df_metrics, df_featsel, df_whitetest], ['Metrics','feature_select','whitetest']):
            if not df.empty:
                df.to_csv(os.path.join(results_path, f'{filename}_full_{timestamp}.csv'),index_label=index_label, float_format=self.df_float_fmt)
        # Store single metrics separately
        for name in self.metr_names['Metrics']:
            columns = [column for column in df_metrics.columns if name in column]
            df_metrics[columns].to_csv(os.path.join(results_path, f'Metrics_{timestamp}_{name}.csv'),index_label=index_label, float_format=self.df_float_fmt)

    def store_metr_df(self, df, output_fname="", index_label=""):
        """
        Store metrics df to file
        @param df: dataframe
        @param output_fname: output filename
        @param index_label: index label
        """
        df.to_csv(output_fname, index_label=index_label, float_format=self.df_float_fmt)