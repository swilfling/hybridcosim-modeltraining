from ModelTraining.datamodels.datamodels.processing.datascaler import DataScaler, Normalizer
from ModelTraining.feature_engineering.expandedmodel import ExpandedModel, TransformerSet
from ModelTraining.feature_engineering.featureexpanders import FeatureExpansion, PolynomialExpansion, IdentityExpander
from ModelTraining.feature_engineering.featureselectors import FeatureSelector
from ModelTraining.datamodels.datamodels import Model, RandomForestRegression


def test_instantiate_transformerset_feat_sel():
    selector_mic = FeatureSelector.from_name('MICThreshold')
    selector_r = FeatureSelector.from_name('RThreshold')

    exp_params = {"degree": 3, "interaction_only": True, "include_bias": False}
    expander = FeatureExpansion.from_name("PolynomialExpansion", **exp_params)
    list_transformers = [selector_mic, expander, selector_r]
    transformer_set = TransformerSet(list_transformers)
    assert(isinstance(transformer_set.get_transformer_by_name('polynomialexpansion'), PolynomialExpansion))


def test_instantiate_expandedmodel():

    model = RandomForestRegression(x_scaler_class=Normalizer, name="RF",  parameters={})
    exp_params = {"degree": 3, "interaction_only": True, "include_bias": False}
    expander = FeatureExpansion.from_name("PolynomialExpansion", **exp_params)
    expander_set = TransformerSet(transformers=[expander])
    expanded_model = ExpandedModel(transformers=expander_set, model=model, feature_names=[])
    estimator = expanded_model.model

    expander_from_model = expanded_model.transformers.get_transformer_by_name('polynomialexpansion')
    assert(isinstance(estimator, RandomForestRegression))
    assert(isinstance(expander_from_model,PolynomialExpansion))
    assert(expander_from_model == expander)


if __name__ == "__main__":
    test_instantiate_expandedmodel()
    test_instantiate_transformerset_feat_sel()


