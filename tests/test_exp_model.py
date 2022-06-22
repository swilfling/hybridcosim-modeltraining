from ModelTraining.datamodels.datamodels.processing.datascaler import DataScaler
from ModelTraining.datamodels.datamodels.wrappers.feature_extension import ExpandedModel, TransformerSet
from ModelTraining.Preprocessing.feature_expanders import FeatureExpansion
from ModelTraining.Preprocessing.feature_selectors import FeatureSelector, FeatureSelectionParams
from ModelTraining.datamodels.datamodels import Model

from functools import reduce
from operator import concat

if __name__ == "__main__":
    model_type = "RandomForestRegression"
    model = Model.from_name(model_type, x_scaler_class=DataScaler.cls_from_name('Normalizer'), name="Model", parameters={})

    exp_params = {"degree": 3, "interaction_only": True, "include_bias": False}
    expansion = ['IdentityExpander', 'PolynomialExpansion']
    expanders = FeatureExpansion.from_names(expansion, **exp_params)
    expander_set = TransformerSet(transformers=expanders)
    expanded_model = ExpandedModel(transformers=expander_set, model=model, feature_names=[])

    estimator = expanded_model.get_estimator()

    selection_params_mic = FeatureSelectionParams(sel_type='MIC-value')
    selector_mic = FeatureSelector.from_params(selection_params_mic)
    selection_params_r = FeatureSelectionParams(sel_type='R-value')
    selector_r = FeatureSelector.from_params(selection_params_r)
    selectors = [selector_mic, selector_r]

    list_transformers = [expanders[0], selector_mic, expanders[1], selector_r]
    transformer_set = TransformerSet(list_transformers)

    list_transformers_red = reduce(concat, [[expander, selector] for expander, selector in zip(expanders, selectors)])