import os

import pandas as pd
import copy
from ModelTraining.Utilities.Parameters import TrainingParams
from ModelTraining.datamodels.datamodels.processing.feature_extension import PolynomialExpansion



def expand_features(data, static_feature_names, target_feature_names,expander_parameters={}):
    expander = PolynomialExpansion(**expander_parameters)
    data_expanded = expander.fit_transform(data[static_feature_names])
    feature_names_expanded = expander.get_feature_names_out(static_feature_names)
    data_outputs = data[target_feature_names]
    data_expanded_df = pd.DataFrame(data_expanded, columns=feature_names_expanded)
    return data_expanded_df.join(data_outputs.set_index(data_expanded_df.index))



############################# Model saving and plotting ##############################

def save_model_and_parameters(results_main_dir, model, training_params: TrainingParams):
    model_dir = os.path.join(results_main_dir, training_params.model_name)
    os.makedirs(model_dir, exist_ok=True)
    # Save model and parameters
    model.save(model_dir)
    training_params.to_file(os.path.join(model_dir, f"parameters_{training_params.model_name}.json"))


def set_train_params_model(training_params_basic_config, feature_set, target_feature, model_type, expansion=None):
    training_params = copy.copy(training_params_basic_config)
    training_params.model_type = model_type
    training_params.model_name = target_feature
    training_params.target_features = [target_feature]
    training_params.static_input_features = feature_set.get_static_input_feature_names(target_feature)
    training_params.dynamic_input_features = feature_set.get_dynamic_input_feature_names(target_feature)
    training_params.dynamic_output_features = feature_set.get_dynamic_output_feature_names(target_feature)
    training_params.expansion = expansion if expansion is not None else training_params.expansion
    return training_params


################################ Misc #######################################################


def add_names_to_features(static_feature_names, static_row):
    return {name: val for name, val in zip(static_feature_names, list(static_row.flatten()))}

