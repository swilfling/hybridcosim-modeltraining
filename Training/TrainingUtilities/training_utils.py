import json
import os

import numpy as np
import pandas as pd
import copy

from sklearn.model_selection import train_test_split

from ...Utilities import feature_combination as fc
from ...Training.TrainingUtilities.parameters import TrainingParams
from ...Training.TrainingUtilities.trainingparams_expanded import TrainingParamsExpanded
from ...datamodels.datamodels.wrappers.expandedmodel import ExpandedModel
from ...feature_engineering.featureengineering.featureexpanders import PolynomialExpansion
from ...datamodels.datamodels.processing.shape import get_windows
from ...datamodels.datamodels import Model
from ...Utilities.trainingdata import TrainingData
from ...Data.DataImport.dataimport import DataImport
from ...Utilities.MetricsExport import ResultExport, MetricsVal, MetricsCalc
from ...feature_engineering.featureengineering.featureselectors import FeatureSelector
from ...datamodels.datamodels.wrappers.expandedmodel import TransformerParams

########################### Data import #################################################


def import_data(cfg_path: str, data_dir:str, dict_usecase):
    """
    Import data using configuration
    :param cfg_path: Path to configuration files for data import
    :param data_dir: path to datasets
    :param dict_usecase: dict containing information about dataset directory and filename
    :return: data
    """
    data_import = DataImport.load(
        os.path.join(cfg_path, dict_usecase['dataset_dir'], f"{dict_usecase['dataset_filename']}.json"))
    data = data_import.import_data(
        os.path.join(data_dir, "Data", dict_usecase['dataset_dir'], dict_usecase['dataset_filename']))
    return data


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
    if isinstance(model, Model):
        model.save(model_dir)
    if isinstance(model, ExpandedModel):
        model.save_pkl(model_dir, "expanded_model.pkl")
    training_params.to_file(os.path.join(model_dir, f"parameters_{training_params.model_name}.json"))


def set_train_params_transformers(training_params: TrainingParamsExpanded, dict_usecase):
    """
    Set training parameters for transformers based on use case
    :param training_params:
    :param dict_usecase:
    :return: modified training params
    """
    tr_params = training_params.transformer_params
    for cfg in TransformerParams.get_params_of_type(tr_params, "CategoricalFeatures"):
        cfg.params['selected_feats'] = dict_usecase.get('cyclical_feats', [])
    for cfg in TransformerParams.get_params_of_type(tr_params, "CyclicFeatures"):
        cfg.params['selected_feats'] = dict_usecase.get('onehot_feats', [])

    for cfg in TransformerParams.get_params_of_type(tr_params, 'Transformer_MaskFeats', 'transformer_params',
                                                         'DynamicFeatures'):
        cfg.params['mask_params']['features_to_transform'] = training_params.dynamic_input_features
    for cfg in TransformerParams.get_params_of_type(tr_params, 'Transformer_MaskFeats', 'transformer_type',
                                                          'StatisticalFeaturesNumpy'):
        cfg.params['mask_params']['features_to_transform'] = dict_usecase['stat_feats']

    for cfg in TransformerParams.get_params_of_type(tr_params, 'Transformer_MaskFeats', 'InverseTransform'):
        cfg.params['mask_params']['features_to_transform'] = dict_usecase.get('to_invert', [])


def set_train_params_model(training_params_basic_config, feature_set, target_feature, model_type, transformer_params=None):
    """
    Set values of training params - Specific for use case with one target feature!
    """
    training_params = copy.copy(training_params_basic_config)
    training_params.model_type = model_type
    training_params.model_name = target_feature
    training_params.target_features = [target_feature]
    training_params.static_input_features = feature_set.get_static_input_feature_names(target_feature)
    training_params.dynamic_input_features = feature_set.get_dynamic_input_feature_names(target_feature)
    training_params.dynamic_output_features = feature_set.get_dynamic_output_feature_names(target_feature)
    if isinstance(training_params_basic_config, TrainingParamsExpanded) and transformer_params is not None:
        training_params.transformer_params = transformer_params
    return training_params


################################ Training and test set #######################################################


def expand_features(data: pd.DataFrame, feature_names, expander_parameters={}):
    """
    Expand features through polynomial expansion
    @param data: input data
    @param feature_names: names of features to expand
    @param expander_parameters: Parameters for polynomial expansion
    @return: dataframe containing expanded data
    """
    expander = PolynomialExpansion(**expander_parameters)
    data_expanded = expander.fit_transform(data[feature_names])
    feature_names_expanded = expander.get_feature_names_out(feature_names)
    return pd.DataFrame(data_expanded, columns=feature_names_expanded)


def split_into_training_and_test_set(index, x, y, training_split=0.8, shuffle=False):
    """
    Split data into training and test set
    @param index: index (n_samples)
    @param x: input data (n_samples, lookback + 1, n_features)
    @param y: target values (n_samples, n_target_features)
    @training_split: fraction of data to use in training set
    @param shuffle: use random sampling
    @return: separated data
    """
    index_train, index_test, x_train, x_test, y_train, y_test = train_test_split(index, x, y, train_size=training_split, shuffle=shuffle)
    return index_train, x_train, y_train, index_test, x_test, y_test


def create_train_data(index, x, y, training_split=0.8, shuffle=False):
    """
    Split data into training and test set
    @param index: index (n_samples)
    @param x: input data (n_samples, lookback + 1, n_features)
    @param y: target values (n_samples, n_target_features)
    @training_split: fraction of data to use in training set
    @param shuffle: use random sampling
    @return: separated data
    """
    index_train, index_test, x_train, x_test, y_train, y_test = train_test_split(index, x, y, train_size=training_split,
                                                                                 shuffle=shuffle)
    return TrainingData(train_index=index_train, test_index=index_test, train_target=y_train, test_target=y_test,
                        train_input=x_train, test_input=x_test)

def extract_training_and_test_set(data: pd.DataFrame, training_params: TrainingParams, create_df=False):
    """
    Extract training and test set
    @param data: full dataset
    @param training_params: training parameters
    @return: index, x, y, feature names
    """
    lookback_horizon = training_params.lookback_horizon
    prediction_horizon = training_params.prediction_horizon

    index = data.iloc[lookback_horizon + prediction_horizon:].index.to_series()

    num_targets = len(training_params.target_features)
    num_samples = data.shape[0]
    targets = np.reshape(data[training_params.target_features].to_numpy(), (num_samples, num_targets))

    """
    DYNAMIC FEATURES
    shape: number of samples, lookback horizon + 1, number of features
    """
    dynamic_feat_names = training_params.dynamic_input_features + training_params.dynamic_output_features
    dynamic_features = data[dynamic_feat_names].to_numpy()
    dynamic_features, y = get_windows(
        features=dynamic_features,
        targets=targets,
        lookback=lookback_horizon,
        lookahead=prediction_horizon
    )
    dynamic_feat_names_lag = [f'{name}_{lag}' for lag in range(1,lookback_horizon+1) for name in dynamic_feat_names]
    dynamic_feature_names_full = dynamic_feat_names + dynamic_feat_names_lag

    static_features = None
    if training_params.static_input_features:
        """
        STATIC FEATURES
        shape: number of samples, 1, number of features
        """
        static_features = data[training_params.static_input_features].to_numpy()
        static_features = static_features[lookback_horizon:static_features.shape[0] - prediction_horizon]
        static_features = np.expand_dims(static_features, axis=1)

        """
        RESHAPED DYNAMIC FEATURES (to fit STATIC FEATURE shape)
        shape: number of samples, 1, (lookback horizon + 1) * number of features)
        """
        dynamic_features = dynamic_features.reshape(
            (dynamic_features.shape[0], 1, dynamic_features.shape[1] * dynamic_features.shape[2]))

    x = fc.combine_static_and_dynamic_features(static_features, dynamic_features)
    feature_names = training_params.static_input_features + dynamic_feature_names_full

    if create_df:
        x = pd.DataFrame(index=index, data=x[:, 0, :], columns=feature_names)
        y = pd.DataFrame(index=index, data=y[:, 0, :], columns=training_params.target_features)
    return index, x, y, feature_names

"""
Replace the ground truth by the predicted values for all the target features in the dataset and save the new one
Assuming that each csv file contains values of one column (each model predicts one target feature without replicas)
"""
def replace_dataset(data, list_training_parameters, first_train_results_path, lookback_horizon):
    new_dataset = data[lookback_horizon + 1:]

    """
    Predicted dataset starts from (lookback horizon + 1)
    """
    for training_parameters in list_training_parameters:

        for feature in training_parameters.target_features:
            val = pd.read_csv(f"{first_train_results_path}/{training_parameters.model_type}_{feature}_predictions.csv", sep=',', engine='python').set_index('date')
            new_dataset = new_dataset.drop(feature, axis=1)
            new_dataset[feature] = val['predicted']

    return new_dataset

################################ Store all results #########################################

def store_results(model,train_data, training_params, metr_exp: MetricsCalc, best_params, result_dir_model, result_id, experiment_name, usecase_name):
    model_dir = os.path.join(result_dir_model, f"Models/{training_params.model_name}/{training_params.model_type}")
    save_model_and_params(model, training_params, model_dir)
    # Calculate and export metrics
    train_data.save_pkl(result_dir_model, "result_data.pkl")
    # Calc metrics
    for k, val in best_params.items():
            metr_exp.add_metr_val(MetricsVal(model_type=model.model.model_type,
                                             model_name=model.name,
                                             featsel_thresh=k.split("__")[0],
                                             expansion_type=result_id,
                                             metrics_type="best_params", metrics_name=k, val=val,
                                             usecase_name=usecase_name))

    metr_vals = metr_exp.calc_all_metrics(train_data, model.transformers.get_transformers_of_type(FeatureSelector))
    # metr_vals = metr_exp.calc_perf_metrics(train_data, model.get_num_predictors())
    # Set metrics identifiers
    transformer_name = model.transformers.get_transformer_by_index(-1).__class__.__name__
    for metr_val in metr_vals:
        metr_val.set_metr_properties(model.model.model_type, model.name, result_id, transformer_name, usecase_name)
    # Store metrics separately
    metr_exp.add_metr_vals(metr_vals)
    for type in ['Metrics', 'pvalues', 'FeatureSelect']:
        filename = os.path.join(result_dir_model, f"{type}_{experiment_name}.csv")
        metr_exp.store_metr_df(metr_exp.get_metr_df(type), filename)
    # Export results
    exp = ResultExport(results_root=result_dir_model, plot_enabled=True)
    exp.export_result(train_data, result_id, show_fig=False)
    exp.plot_enabled = False
    exp.export_model_properties(model, result_id)
    exp.export_featsel_metrs(model, result_id)

########################## Load from JSON ########################################

def load_from_json(filename):
    with open(filename) as f:
        return json.load(f)