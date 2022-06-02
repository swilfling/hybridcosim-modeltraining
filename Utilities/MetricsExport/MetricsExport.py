import os
import pandas as pd
from ...datamodels.datamodels import Model
from ...datamodels.datamodels.validation import metrics
from ...datamodels.datamodels.validation.white_test import white_test
from ...Utilities.Plotting import plotting_utilities as plt_utils
from . import metr_utils
from ..Parameters.trainingresults import TrainingResults


class MetricsExport:
    """
    MetricsExport
    Export metrics of models
    """
    plot_enabled: bool = False
    results_root: str = "./"
    result_dir = 'Results'
    scatter_dir = 'Scatter'
    tseries_dir = 'Timeseries'
    featsel_dir = 'FeatureSelection'
    property_dir = 'ModelProperties'
    metr_names = {'Metrics': ['R2', 'CV-RMS', 'MAPE'],
                     'FeatureSelect': ['selected_features', 'all_features'],
                     'pvalues': ['pvalue_lm']}

    def __init__(self, results_root="./", plot_enabled=False,**kwargs):
        self.plot_enabled = plot_enabled
        self.results_root = results_root
        if kwargs.get('metr_names', False):
            self.metr_names = kwargs.get('metr_names')

    ############################################### Directories #######################################################

    def create_dirs(self):
        plot_dir = os.path.join(self.results_root, self.result_dir)
        os.makedirs(plot_dir, exist_ok=True)
        if self.plot_enabled:
            scatter_plot_dir = os.path.join(plot_dir, self.scatter_dir)
            ts_plot_dir = os.path.join(plot_dir, self.tseries_dir)
            os.makedirs(scatter_plot_dir, exist_ok=True)
            os.makedirs(ts_plot_dir, exist_ok=True)

    def get_scatter_dir(self):
        return os.path.join(self.results_root, self.result_dir, self.scatter_dir)

    def get_tseries_dir(self):
        return os.path.join(self.results_root, self.result_dir, self.tseries_dir)

    def get_result_dir(self):
        return os.path.join(self.results_root, self.result_dir)

    ############################################## Export functions ####################################################

    def export_coeffs(self, coeffs, feature_names, dir, title, ylabel=""):
        """
        Export coefficient values - optional: barplot
        @param coeffs: array of coefficients
        @param feature_names: feature names
        @param dir: output directory
        @param title: file title
        @param ylabel: ylabel for plotting
        """
        df = pd.DataFrame(data=coeffs, index=feature_names)
        df.to_csv(os.path.join(dir, f'Coefficients_{title}.csv'), float_format='%.2f', index_label='Feature')
        if self.plot_enabled:
            for col in df.columns:
                plt_utils.barplot(df.index, df[col].to_numpy(), dir, f'Coefficients_{title}', ylabel=ylabel,
                                  figsize=(30, 7))

    def export_featsel_metrs(self, expanders, selectors, feature_names):
        """
        Export feature selection metrics - Combination of expanders and selectors
        @param expanders: Expanders - e.g. PolynomialExpansion
        @param selectors: Feature selectors
        @param feature_names: basic input feature names
        """
        output_dir = os.path.join(self.results_root, self.featsel_dir)
        os.makedirs(output_dir, exist_ok=True)
        for i, selector in enumerate(selectors):
            selector.save_pkl(output_dir, f"{selector.__class__.__name__}_{i}.pickle")
        for expander, selector in zip(expanders, selectors):
            feature_names = expander.get_feature_names_out(feature_names)
            self.export_coeffs(selector.get_coef(), feature_names=feature_names, dir=output_dir,
                               title=f'{selector.__class__.__name__} - {expander.__class__.__name__}',
                               ylabel=str(selector.__class__.__name__))

    def export_model_properties(self, model: Model):
        """
        Export model properties - F-score or linear regression coeffs
        @param model: model to export
        """
        property_dir = os.path.join(self.results_root, self.property_dir)
        os.makedirs(property_dir, exist_ok=True)
        model_type = model.__class__.__name__
        title = f'{model_type}_{model.name}_{model.expanders[-2].__class__.__name__}'
        # Export coefficients
        if model_type in ['RandomForestRegression', 'RidgeRegression', 'LinearRegression']:
            ylabel = 'F-Score' if model_type == 'RandomForestRegression' else 'Coefficients'
            self.export_coeffs(model.get_coef().T, model.get_expanded_feature_names(), property_dir, title, ylabel)
        if model_type == 'SymbolicRegression':
            metr_utils.metrics_to_json({'Program': str(model.get_program())}, os.path.join(property_dir, f'Program_{title}.json'))
        if model_type == 'RuleFitRegression':
            model.get_rules().to_csv(os.path.join(property_dir, f'Rules_{title}.csv'), float_format="%.2f", index_label='Rule')

    def export_result(self, result: TrainingResults, title=""):
        """
        Export result to csv - optional plotting
        @param result: TrainingResults structure
        @param title: Optional: title
        """
        result.test_results_to_csv(self.get_result_dir(), f'Timeseries_{title}.csv')
        if self.plot_enabled:
            for feat in result.target_feat_names:
                plt_utils.plot_result(result.test_result_df(feat), self.get_tseries_dir(),
                                      f"Timeseries_{feat}_{title}", store_to_csv=False, figsize=(14, 4))
                plt_utils.scatterplot(result.test_pred_vals(feat), result.test_target_vals(feat), self.get_scatter_dir(),
                                      f"Scatter_{feat}_{title}")

    ################################################ Analysis Functions ################################################

    def analyze_result(self, model: Model, result: TrainingResults):
        """
        Analyze result
        Performs
            - Metrics computation
            - White test for heteroscedascity
            - Export model properties and result data
        @param model: Model to analyze
        @param result: TraniningResults
        Returns:
            @return pd.DataFrame containing metrics and white test
        """
        df_metr_full = pd.DataFrame()
        model_name = f'{model.name}_{model.expanders[-2].__class__.__name__}'
        model_name_full = f'{model.__class__.__name__}_{model_name}'
        # Perform White test
        df_white = self.white_test_allfeats(result).add_suffix(f'_{model_name}_pvalues')
        df_metr_full = df_white if df_metr_full.empty else df_metr_full.join(df_white)
        # Calculate metrics
        df_metr = self.calc_metrics_allfeats(result, len(model.get_expanded_feature_names())).add_suffix(f'_{model_name}_pvalues')
        df_metr_full = df_metr if df_metr_full.empty else df_metr_full.join(df_metr)
        # Export model properties
        self.export_model_properties(model)
        self.export_result(result, model_name_full)
        df_metr_full.index = [model.__class__.__name__]
        return df_metr_full

    def analyze_results_full(self, models, results, list_selectors=[]):
        """
        Analyze result
        Performs
            - Feature selection analysis
            - Metrics computation
            - White test for heteroscedascity
            - Export model properties and result data
        @param model: Model to analyze
        @param result: TraniningResults
        @param list_selectors: List of lists of feature selectors
        Returns:
            @return: pd.DataFrame containing metrics, feature selection and white test
        """
        df_metr_full = pd.DataFrame()
        # Export feature selection metrics
        for model, selectors in zip(models, list_selectors):
            self.export_featsel_metrs(model.expanders, selectors, model.feature_names)
            df_featsel = self.analyze_featsel(model, selectors)
            df_metr_full = df_featsel if df_metr_full.empty else df_metr_full.join(df_featsel)

        self.create_dirs()
        for model, result in zip(models, results):
            df_metr = self.analyze_result(model, result)
            df_metr_full = df_metr if df_metr_full.empty else df_metr_full.join(df_metr)
        df_metr_full.index = [models[0].__class__.__name__]
        return df_metr_full

    ################################# Metrics calculation #############################################################

    def analyze_featsel(self, model, selectors):
        """
        Analyze feature selection
        @param model: Model to analyze - expanders are needed to get feature names
        @param selectors: List of selectors
        @return: pd.DataFrame containing results
        """
        model_name = f'{model.name}_{model.expanders[-2].__class__.__name__}'
        df_featsel_full = pd.DataFrame()
        for i, selector in enumerate(selectors):
            sel_metrs = selector.get_metrics()
            selector_name = f'{selector.__class__.__name__}_{i}'
            df_featsel = pd.DataFrame(data=sel_metrs.values(), index=sel_metrs.keys()).transpose()
            df_featsel = df_featsel[self.metr_names['FeatureSelect']].add_suffix(f'_{selector_name}_{model_name}_FeatureSelect')
            df_featsel_full = df_featsel if df_featsel_full.empty else df_featsel_full.join(df_featsel)
        df_featsel_full.index = [model.__class__.__name__]
        return df_featsel_full

    def white_test_allfeats(self, result: TrainingResults):
        """
        White test for all target features in the training result.
        @param result: TrainingResult structure
        @param selected_metrics: Selected metrics for white test. Supported: 'pvalue_lm', 'pvalue_f'
        """
        df_white = pd.DataFrame()
        for feat in result.target_feat_names:
            white_pvals = white_test(result.test_input, result.test_target_vals(feat) - result.test_pred_vals(feat))
            df_white_feat = pd.DataFrame(data=white_pvals.values(), index=white_pvals.keys()).transpose()[self.metr_names['pvalues']].add_suffix(f'_{feat}')
            df_white = df_white_feat if df_white.empty else df_white.join(df_white_feat)
        return df_white

    def calc_metrics_allfeats(self, result: TrainingResults, n_predictors=0):
        """
        Calculate metrics
        @param result: TrainingResults structure
        @param num_predictors: number of predictors
        @param metr_names: metrics names
        @return pd.Dataframe containing metrics
        """
        n_samples = result.train_index.shape[0] + result.test_index.shape[0]
        df_metrs = pd.DataFrame()
        for feat in result.target_feat_names:
            y_true = result.test_target_vals(feat)
            y_pred = result.test_pred_vals(feat)
            # Get metrics
            metrs = metrics.all_metrics(y_true=y_true, y_pred=y_pred)
            # Select metrics
            metrs = {k: v for k, v in metrs.items() if k in self.metr_names['Metrics']}
            # Additional metrics
            if 'RA' in self.metr_names['Metrics']:
                metrs.update({"RA": metrics.rsquared_adj(y_true, y_pred, n_samples, n_predictors)})
            if 'RA_SKLEARN' in self.metr_names['Metrics']:
                metrs.update(
                    {"RA_SKLEARN": metrics.rsquared_sklearn_adj(y_true, y_pred, n_samples, n_predictors)})
            df_metrs_feat = pd.DataFrame(data=metrs.values(), index=metrs.keys()).transpose()[self.metr_names['Metrics']].add_suffix(f'_{feat}')
            df_metrs = df_metrs_feat if df_metrs.empty else df_metrs.join(df_metrs_feat)
        return df_metrs

    ############################################# Store metrics ########################################################

    def store_all_metrics(self, df_full:pd.DataFrame, results_path="", timestamp=""):
        """
        Store metrics df to files - get subsets of dataframes to store separately
        @param df_full: metrics dataframe
        @param results_path: result dir - optional argument
        @param timestamp: timestamp of experiment
        """
        results_path = self.results_root if results_path == "" else results_path
        df_full.to_csv(os.path.join(results_path, f'AllMetrics_full_{timestamp}.csv'),index_label='Model', float_format='%.3f')

        # Get subsets
        df_metrics = metr_utils.get_df_subset(df_full, 'Metrics')
        df_featsel = metr_utils.get_df_subset(df_full, 'FeatureSelect')
        df_whitetest = metr_utils.get_df_subset(df_full, 'pvalues')

        # Store results
        for df, filename in zip([df_metrics, df_featsel, df_whitetest], ['Metrics','feature_select','whitetest']):
            if not df.empty:
                df.to_csv(os.path.join(results_path, f'{filename}_full_{timestamp}.csv'),index_label='Model', float_format='%.3f')
        # Store single metrics separately
        for name in self.metr_names['Metrics']:
            columns = [column for column in df_metrics.columns if name in column]
            df_metrics[columns].to_csv(os.path.join(results_path, f'Metrics_{timestamp}_{name}.csv'),index_label='Model', float_format='%.3f')
