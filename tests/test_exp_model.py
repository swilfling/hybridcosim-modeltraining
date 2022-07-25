from ModelTraining.datamodels.datamodels.processing.datascaler import Normalizer
from ModelTraining.feature_engineering.expandedmodel import ExpandedModel, TransformerSet
from ModelTraining.feature_engineering.featureengineeringbasic.featureexpanders import FeatureExpansion, PolynomialExpansion
from ModelTraining.feature_engineering.featureengineeringbasic.featureselectors import FeatureSelector
from ModelTraining.datamodels.datamodels import LinearRegression
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

    model = LinearRegression(x_scaler_class=Normalizer, name="RF",  parameters={})
    exp_params = {"degree": 3, "interaction_only": True, "include_bias": False}
    expander = FeatureExpansion.from_name("PolynomialExpansion", **exp_params)
    expander_set = TransformerSet(transformers=[expander])
    expanded_model = ExpandedModel(transformers=expander_set, model=model, feature_names=[])
    estimator = expanded_model.model

    expander_from_model = expanded_model.transformers.get_transformer_by_name('polynomialexpansion')
    assert(isinstance(estimator, LinearRegression))
    assert(isinstance(expander_from_model,PolynomialExpansion))
    assert(expander_from_model == expander)


def test_expandedmodel():
    x_train = np.random.randn(100,1,5)
    y_train = np.random.randn(100,1,1)

    x_test = np.random.randn(20, 1, 5)

    model = LinearRegression(x_scaler_class=Normalizer, name="RF", parameters={})
    exp_params = {"degree": 3, "interaction_only": True, "include_bias": False}
    expander = FeatureExpansion.from_name("PolynomialExpansion", **exp_params)
    expander_set = TransformerSet(transformers=[expander])
    expanded_model = ExpandedModel(transformers=expander_set, model=model, feature_names=[])

    expanded_model.train(x_train, y_train)
    y_pr_1 = expanded_model.predict(x_test)

    x_tr = np.expand_dims(expander.fit_transform(x_train,y_train),axis=1)
    model_2 = LinearRegression(x_scaler_class=Normalizer, name="RF", parameters={})
    model_2.train(x_tr,y_train)
    x_test_tr = np.expand_dims(expander.transform(x_test),axis=1)
    y_pr_2 = model_2.predict(x_test_tr)

    assert(np.all(np.isclose(y_pr_1.flatten(), y_pr_2.flatten())))


if __name__ == "__main__":
    test_instantiate_expandedmodel()
    test_instantiate_transformerset_feat_sel()
    test_expandedmodel()

