from ModelTraining.datamodels.datamodels.processing.datascaler import DataScaler
from ModelTraining.feature_engineering.expandedmodel import ExpandedModel, TransformerSet
from ModelTraining.feature_engineering.feature_expanders import FeatureExpansion
from ModelTraining.feature_engineering.feature_selectors import FeatureSelector
from ModelTraining.datamodels.datamodels import Model

from functools import reduce
from operator import concat

if __name__ == "__main__":
    model_type = "RandomForestRegression"
    model = Model.from_name(model_type, x_scaler_class=DataScaler.cls_from_name('Normalizer'), name="Model", parameters={})

    exp_params = {"degree": 3, "interaction_only": True, "include_bias": False}
    expanders = [FeatureExpansion.from_name(name, **exp_params) for name in ['IdentityExpander', 'PolynomialExpansion']]
    expander_set = TransformerSet(transformers=expanders)
    expanded_model = ExpandedModel(transformers=expander_set, model=model, feature_names=[])

    estimator = expanded_model.get_estimator()

    selector_mic = FeatureSelector.from_name('MICThreshold')
    selector_r = FeatureSelector.from_name('RThreshold')
    selectors = [selector_mic, selector_r]

    list_transformers = [expanders[0], selector_mic, expanders[1], selector_r]
    transformer_set = TransformerSet(list_transformers)

    list_transformers_red = reduce(concat, [[expander, selector] for expander, selector in zip(expanders, selectors)])