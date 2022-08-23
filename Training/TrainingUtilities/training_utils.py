import json
import os
import copy

from ...Training.TrainingUtilities.parameters import TrainingParams


############################# Model saving and plotting ##############################

def save_model_and_params(model, training_params: TrainingParams, results_main_dir: str="./"):
    """
    Save model and training parameters
    @param results_main_dir: output directory
    @param model: Model to save
    @param training_params: training parameters
    """
    model_dir = os.path.join(results_main_dir, training_params.model_name)
    os.makedirs(model_dir, exist_ok=True)
    # Save model and parameters
    model.save(model_dir)
    training_params.to_file(os.path.join(model_dir, f"parameters_{training_params.model_name}.json"))


def set_train_params_model(training_params_basic_config: TrainingParams, feature_set, target_feature, model_type):
    """
    Set values of training params - Specific for use case with one target feature!
    @param training_params_basic_config: basic setup for training params
    @param feature_set: FeatureSet containing input and outputs
    @param target_feature: Selected target
    @param model_type: model type, e.g. linear regression
    @return: TrainingParams object with modified parameters
    """
    training_params = copy.copy(training_params_basic_config)
    training_params.model_type = model_type
    training_params.model_name = target_feature
    training_params.target_features = [target_feature]
    training_params.static_input_features = feature_set.get_static_input_feature_names(target_feature)
    training_params.dynamic_input_features = feature_set.get_dynamic_input_feature_names(target_feature)
    training_params.dynamic_output_features = feature_set.get_dynamic_output_feature_names(target_feature)
    return training_params


########################## Load from JSON ########################################

def load_from_json(filename):
    with open(filename) as f:
        return json.load(f)