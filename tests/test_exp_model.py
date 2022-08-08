from ModelTraining.datamodels.datamodels.processing.datascaler import Normalizer
from ModelTraining.feature_engineering.expandedmodel import ExpandedModel, TransformerSet
from ModelTraining.feature_engineering.featureengineering.featureexpanders import FeatureExpansion, PolynomialExpansion
from ModelTraining.feature_engineering.featureengineering.featureselectors import FeatureSelector
from ModelTraining.datamodels.datamodels import RandomForestRegression, LinearRegression
import numpy as np


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


def test_expandedmodel_data():
    data = np.random.randn(100,1,3)
    target = np.random.randn(100,1,1)
    model_1 = LinearRegression(x_scaler_class=Normalizer, name="RF", parameters={})
    model_2 = LinearRegression(x_scaler_class=Normalizer, name="RF", parameters={})
    exp_params = {"degree": 3, "interaction_only": True, "include_bias": False}
    expander = FeatureExpansion.from_name("PolynomialExpansion", **exp_params)
    expander_set = TransformerSet(transformers=[expander])
    expanded_model = ExpandedModel(transformers=expander_set, model=model_2, feature_names=[])

    # Check transformation
    data_transformed = expander.fit_transform(data)
    data_transformed_exp_model = expanded_model.transform_features(data)
    assert(np.all(data_transformed == data_transformed_exp_model))

    # Check training
    model_1.train(np.expand_dims(data_transformed,axis=1), target)
    expanded_model.train(data, target)
    assert(np.all(np.isclose(model_1.model.coef_, expanded_model.model.model.coef_)))



if __name__ == "__main__":
    test_instantiate_expandedmodel()
    test_instantiate_transformerset_feat_sel()
    test_expandedmodel_data()

