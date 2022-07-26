import os
import pandas as pd

from ModelTraining.feature_engineering.featureengineering.featureselectors.thresholds import FeatureSelectThreshold
from . import metr_utils
from ..trainingdata import TrainingData
from ..Plotting import plot_data as plt_utils
from ...feature_engineering.expandedmodel import ExpandedModel
from ...datamodels.datamodels import LinearModel, RandomForestRegression, SymbolicRegression, RuleFitRegression


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

    def export_result_full(self, model: ExpandedModel, result: TrainingData, title=""):
        """
        Export all results.
        @param model: Model
        @param result: TrainingData
        @param title: title for output files
        """
        model_name = f'{model.name}_{model.transformers.type_last_transf()}'
        model_name_full = f'{model.model.__class__.__name__}_{model_name}'
        self.export_featsel_metrs(model, title)
        self.export_model_properties(model, title)
        self.export_result(result, f'{model_name_full}_{title}')

    ############################################## Export functions ####################################################

    def export_coeffs(self, coeffs, feature_names, dir, title="", ylabel=""):
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

    def export_featsel_metrs(self, model: ExpandedModel, title=""):
        """
        Export feature selection metrics - Combination of expanders and selectors
        @param model: ExpandedModel
        """
        output_dir = os.path.join(self.results_root, self.featsel_dir)
        os.makedirs(output_dir, exist_ok=True)
        transformers = model.transformers.get_list_transfomers()
        feature_names = model.feature_names
        for transformer in transformers:
            if isinstance(transformer, FeatureSelectThreshold):
                self.export_coeffs(transformer.get_coef(), feature_names=transformer.get_feature_names_out(feature_names), dir=output_dir,
                               title=f'{transformer.__class__.__name__} - {title}',
                               ylabel=str(transformer.__class__.__name__))
            feature_names = transformer.get_feature_names_out(feature_names)

    def export_model_properties(self, model: ExpandedModel, title=""):
        """
        Export model properties - F-score or linear regression coeffs
        @param model: model to export
        """
        property_dir = os.path.join(self.results_root, self.property_dir)
        os.makedirs(property_dir, exist_ok=True)
        regressor = model.model
        title_full = f'{regressor.__class__.__name__}_{model.name}_{title}'
        # Export coefficients
        if isinstance(regressor, LinearModel):
            self.export_coeffs(regressor.get_coef().T, model.get_transformed_feature_names(), property_dir,
                               title_full, "Coefficients")
        if isinstance(regressor, RandomForestRegression):
            self.export_coeffs(regressor.get_coef().T, model.get_transformed_feature_names(), property_dir,
                               title_full, "F-Score")
        if isinstance(regressor, SymbolicRegression):
            metr_utils.dict_to_json({'Program': str(regressor.get_program())},
                                    os.path.join(property_dir, f'Program_{title_full}.json'))
        if isinstance(regressor, RuleFitRegression):
            regressor.get_rules().to_csv(os.path.join(property_dir, f'Rules_{title_full}.csv'),
                                                float_format="%.2f", index_label='Rule')

    def export_result(self, result: TrainingData, title="", show_fig=True):
        """
        Export result to csv - optional plotting
        @param result: TrainingResults structure
        @param title: Optional: title
        """
        result.test_results_to_csv(self.results_root, f'TestResults_{title}.csv')
        if self.plot_enabled:
            for feat in result.target_feat_names:
                plt_utils.plot_data(result.test_result_df(feat), self.get_tseries_dir(), filename=f"Timeseries_{feat}_{title}",
                                    store_to_csv=True, figsize=(14, 4), fig_title=f"Timeseries - {feat} - {title}", show_fig=show_fig)
                plt_utils.scatterplot(result.test_pred_vals(feat), result.test_target_vals(feat), self.get_scatter_dir(),
                                      f"Scatter_{feat}_{title}", show_fig=show_fig)
