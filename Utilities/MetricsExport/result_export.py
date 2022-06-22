import os
from typing import List
import pandas as pd

from ModelTraining.feature_engineering.feature_selectors import FeatureSelector
from ModelTraining.feature_engineering.feature_expanders import FeatureExpansion
from . import metr_utils
from ..trainingresults import TrainingResults
from ..Plotting import plot_data as plt_utils
from ModelTraining.feature_engineering.expandedmodel import ExpandedModel


class ResultExport:
    """
    Export results of model training
    """
    plot_enabled: bool = False
    results_root: str = "./"
    plot_dir = 'Plots'
    scatter_dir = 'Scatter'
    tseries_dir = 'Timeseries'
    featsel_dir = 'FeatureSelection'
    property_dir = 'ModelProperties'

    def __init__(self, results_root="./", plot_enabled=False, **kwargs):
        self.plot_enabled = plot_enabled
        self.results_root = results_root
        self.create_dirs()

    ############################################### Directories #######################################################

    def create_dirs(self):
        """
        Create result directory structure.
        """
        plot_dir = os.path.join(self.results_root, self.plot_dir)
        os.makedirs(plot_dir, exist_ok=True)
        if self.plot_enabled:
            scatter_plot_dir = os.path.join(plot_dir, self.scatter_dir)
            ts_plot_dir = os.path.join(plot_dir, self.tseries_dir)
            os.makedirs(scatter_plot_dir, exist_ok=True)
            os.makedirs(ts_plot_dir, exist_ok=True)

    def get_scatter_dir(self):
        return os.path.join(self.results_root, self.plot_dir, self.scatter_dir)

    def get_tseries_dir(self):
        return os.path.join(self.results_root, self.plot_dir, self.tseries_dir)

    def get_plot_dir(self):
        return os.path.join(self.results_root, self.plot_dir)

    ########################################### Export all results #####################################################

    def export_results_full(self, models: List[ExpandedModel], results: List[TrainingResults], list_selectors: List[List[FeatureSelector]]=[]):
        """
        Export all results.
        @param models: List of models
        @param results: List of results - corresponding to models
        @param list_selectors: List of selector sets
        """
        for model, result in zip(models, results):
            model_name = f'{model.name}_{model.transformers.type_last_transf()}'
            model_name_full = f'{model.model.__class__.__name__}_{model_name}'
            self.export_featsel_metrs(model)
            self.export_model_properties(model)
            self.export_result(result, model_name_full)

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
                plt_utils.barplot(df[col], dir, filename=f'Coefficients_{title}', fig_title=f"Coefficients - {title}",
                                                                   ylabel=ylabel, figsize=(30, 7))

    def export_featsel_metrs(self, model: ExpandedModel):
        """
        Export feature selection metrics - Combination of expanders and selectors
        @param model: ExpandedModel
        """
        output_dir = os.path.join(self.results_root, self.featsel_dir)
        os.makedirs(output_dir, exist_ok=True)

        expanders = model.transformers.get_transformers_of_type(FeatureExpansion)
        selectors = model.transformers.get_transformers_of_type(FeatureSelector)
        for expander, selector in zip(expanders, selectors):
            feature_names = expander.get_feature_names_out(model.feature_names)
            self.export_coeffs(selector.get_coef(), feature_names=selector.get_feature_names_out(feature_names), dir=output_dir,
                               title=f'{selector.__class__.__name__} - {expander.__class__.__name__}',
                               ylabel=str(selector.__class__.__name__))

    def export_model_properties(self, model: ExpandedModel):
        """
        Export model properties - F-score or linear regression coeffs
        @param model: model to export
        """
        property_dir = os.path.join(self.results_root, self.property_dir)
        os.makedirs(property_dir, exist_ok=True)
        model_type = model.model.__class__.__name__
        title = f'{model_type}_{model.name}_{model.transformers.type_transf_full()}'
        # Export coefficients
        if model_type in ['RandomForestRegression', 'RidgeRegression', 'LinearRegression']:
            ylabel = 'F-Score' if model_type == 'RandomForestRegression' else 'Coefficients'
            self.export_coeffs(model.model.get_coef().T, model.get_transformed_feature_names(), property_dir, title, ylabel)
        if model_type == 'SymbolicRegression':
            metr_utils.dict_to_json({'Program': str(model.model.get_program())},
                                    os.path.join(property_dir, f'Program_{title}.json'))
        if model_type == 'RuleFitRegression':
            model.model.get_rules().to_csv(os.path.join(property_dir, f'Rules_{title}.csv'), float_format="%.2f",
                                     index_label='Rule')

    def export_result(self, result: TrainingResults, title=""):
        """
        Export result to csv - optional plotting
        @param result: TrainingResults structure
        @param title: Optional: title
        """
        result.test_results_to_csv(self.results_root, f'TestResults_{title}.csv')
        if self.plot_enabled:
            for feat in result.target_feat_names:
                plt_utils.plot_data(result.test_result_df(feat), self.get_tseries_dir(), filename=f"Timeseries_{feat}_{title}",
                                    store_to_csv=False, figsize=(14, 4), fig_title=f"Timeseries - {feat} - {title}")
                plt_utils.scatterplot(result.test_pred_vals(feat), result.test_target_vals(feat), self.get_scatter_dir(),
                                      f"Scatter_{feat}_{title}")
