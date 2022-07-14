from ModelTraining.feature_engineering.parameters import TrainingParamsExpanded, TransformerParams
from ModelTraining.dataimport.data_import import load_from_json
import os


def test_store_load_params():
    # Load basic config
    expander_parameters = load_from_json(
        os.path.join("../", 'Configuration', 'expander_params_PolynomialExpansion.json'))
    train_params = TrainingParamsExpanded.load(
        "../Configuration/TrainingParameters/training_params_poly_normalized.json")
    # Adapt config
    transformer_params_poly = [TransformerParams(type='MICThreshold', params={'thresh': 0.05}),
                               TransformerParams(type='PolynomialExpansion', params=expander_parameters),
                               TransformerParams(type='RThreshold', params={'thresh': 0.05})]
    train_params.transformer_params = transformer_params_poly
    # Store
    train_params.to_file("../Configuration/TrainingParameters/train_params_mic_0_05_poly_r_0_05.json")
    # Load
    train_params_2 = TrainingParamsExpanded.load(
        "../Configuration/TrainingParameters/train_params_mic_0_05_poly_r_0_05.json")
    assert (train_params == train_params_2)



if __name__ == "__main__":
    expander_parameters = load_from_json(os.path.join("../", 'Configuration','expander_params_PolynomialExpansion.json' ))
    transformer_params_basic = [TransformerParams(type='MICThreshold', params={'thresh': 0.05}),
                          TransformerParams(type='IdentityExpander'),
                          TransformerParams(type='RThreshold', params={'thresh': 0.05})]

    transformer_params_poly = [TransformerParams(type='MICThreshold', params={'thresh': 0.05}),
     TransformerParams(type='PolynomialExpansion', params=expander_parameters),
     TransformerParams(type='RThreshold', params={'thresh': 0.05})]

    train_params = TrainingParamsExpanded.load("../Configuration/TrainingParameters/training_params_poly_normalized.json")
    train_params.transformer_params = transformer_params_basic
    train_params.to_file("../Configuration/TrainingParameters/train_params_mic_0_05_basic_r_0_05.json")
    train_params.transformer_params = transformer_params_poly
    train_params.to_file("../Configuration/TrainingParameters/train_params_mic_0_05_poly_r_0_05.json")

    train_params_2 = TrainingParamsExpanded.load(
        "../Configuration/TrainingParameters/train_params_mic_0_05_poly_r_0_05.json")
    assert(train_params == train_params_2)

