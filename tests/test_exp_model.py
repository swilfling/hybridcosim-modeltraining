from ModelTraining.datamodels.datamodels.processing.datascaler import DataScaler
from ModelTraining.datamodels.datamodels.wrappers.expandedmodel import ExpandedModel
from ModelTraining.datamodels.datamodels.processing.feature_extension import ExpanderSet
from ModelTraining.datamodels.datamodels import Model

if __name__ == "__main__":
    model_type = "RandomForestRegression"
    model = Model.from_name(model_type, x_scaler_class=DataScaler.cls_from_name('Normalizer'), name="Model", parameters={})

    exp_params = {"degree": 3, "interaction_only": True, "include_bias": False}
    expansion = ['IdentityExpander', 'PolynomialExpansion']
    expander_set = ExpanderSet.from_names(expansion, **exp_params)
    expanded_model = ExpandedModel(expanders=expander_set, model=model, feature_names=[])

    estimator = expanded_model.get_estimator()