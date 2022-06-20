#%%

import ModelTraining.Preprocessing.FeatureCreation.add_features as feat_utils
from ModelTraining.Preprocessing.featureset import FeatureSet
import ModelTraining.Preprocessing.get_data_and_feature_set
import ModelTraining.Preprocessing.data_analysis as data_analysis
import ModelTraining.Training.TrainingUtilities.training_utils as train_utils
import ModelTraining.Preprocessing.DataImport.data_import as data_import
import ModelTraining.Utilities.Plotting.plot_distributions as plt_dist
import os
import numpy as np
import pandas as pd
from ModelTraining.datamodels.datamodels.processing.datascaler import Normalizer


if __name__ == '__main__':
    #%%
    root_dir = "../"
    data_dir = "../../"
    # Added: Preprocessing - Smooth features
    usecase_config_path = os.path.join(root_dir, 'Configuration/UseCaseConfig')
    list_usecases = ['CPS-Data', 'SensorA6', 'SensorB2', 'SensorC6', 'Solarhouse1','Solarhouse2']

    dict_usecases = [data_import.load_from_json(os.path.join(usecase_config_path, f"{name}.json")) for name in
                     list_usecases]

    interaction_only=True
    matrix_path = "./Figures/Correlation"
    vif_path = './Figures/Correlation/VIF'
    os.makedirs(vif_path, exist_ok=True)
    float_format="%.2f"
    expander_parameters = {'degree': 2, 'interaction_only': True, 'include_bias': False}

    #%%
    ##### Sparsity
    sparsity_dir ="./Figures/Sparsity"
    os.makedirs(sparsity_dir, exist_ok=True)

    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        # Get data and feature set
        data = get_data(os.path.join(data_dir, dict_usecase['dataset']))
        data = feat_utils.add_features_to_data(data, dict_usecase)
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)
        data = data.astype('float')
        # Data preprocessing
        #data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], do_smoothe=True)

        features_for_corrmatrix = [feature.name for feature in feature_set.get_input_feats() if
                                   not feature.cyclic and not feature.statistical]
        if usecase_name in ['CPS-Data', 'SensorA6', 'SensorB2', 'SensorC6']:
            features_for_corrmatrix.remove("holiday_weekend")
            features_for_corrmatrix.append("holiday")

        cur_data = data[features_for_corrmatrix]
        cur_data = cur_data.dropna(axis=0)
        if usecase_name == "Solarhouse1":
            cur_data['SGlobal'][cur_data['SGlobal'] < 30] = 0
        sparsity = np.array([data_analysis.calc_sparsity_abs(cur_data[feature])for feature in features_for_corrmatrix])
        df_sparsity = pd.DataFrame(data=[sparsity], columns=features_for_corrmatrix, index=[usecase_name])
        df_sparsity.to_csv(os.path.join(sparsity_dir, f"Sparsity_{usecase_name}.csv"), float_format="%.2f",
                           index_label="Dataset")
        df_sparsity_percent = df_sparsity * 100
        df_sparsity_percent.to_csv(os.path.join(sparsity_dir, f"Sparsity_{usecase_name}_percent.csv"), float_format="%.2f",
                           index_label="Dataset")

        cur_data = cur_data.dropna(axis=0)
        expanded_features = train_utils.expand_features(cur_data, features_for_corrmatrix, expander_parameters)

        sparsity_exp = np.array([data_analysis.calc_sparsity_abs(expanded_features[feature]) for feature in expanded_features.columns])
        df_sparsity_exp = pd.DataFrame(data=[sparsity_exp], columns=expanded_features.columns, index=[usecase_name])
        df_sparsity_exp.to_csv(os.path.join(sparsity_dir, f"Sparsity_{usecase_name}_expanded.csv"), float_format="%.2f",
                           index_label="Dataset")
        df_sparsity_exp_percent = df_sparsity_exp * 100
        df_sparsity_exp.to_csv(os.path.join(sparsity_dir, f"Sparsity_{usecase_name}_expanded_percent.csv"), float_format="%.2f",
                           index_label="Dataset")

    #%%
    ##### Density
    density_dir ="./Figures/Density"
    os.makedirs(density_dir, exist_ok=True)

    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        density_dir_usecase = os.path.join(density_dir, usecase_name)
        os.makedirs(density_dir_usecase, exist_ok=True)

        # Get data and feature set
        ddata = get_data(os.path.join(data_dir, dict_usecase['dataset']))
        data = feat_utils.add_features_to_data(data, dict_usecase)
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)
        data = data.astype('float')
        # Data preprocessing
        #data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], do_smoothe=True)

        feats_for_density = [feature.name for feature in feature_set.get_input_feats() if
                             not feature.cyclic and not feature.statistical]
        if usecase_name in ['CPS-Data', 'SensorA6', 'SensorB2', 'SensorC6']:
            feats_for_density.remove("holiday_weekend")
            feats_for_density.remove('daylight')
        if usecase_name == 'Solarhouse1':
            feats_for_density.remove('VDSolar_inv')
        if usecase_name == 'Solarhouse1':
            data['SGlobal'][data['SGlobal'] < 30] = 0
        if usecase_name == 'Solarhouse2':
            feats_for_density.remove('Vd_Solar_inv')

        feats_for_density_full = feats_for_density + feature_set.get_output_feature_names()

        scaler = Normalizer()
        scaler.fit(data)
        data = scaler.transform(data)

        fig_title = f'Density - {usecase_name} - nonzero samples'
        plt_dist.plot_density(data[feats_for_density], density_dir_usecase, filename=fig_title,
                              fig_title=fig_title, omit_zero_samples=True, store_tikz=True)
        fig_title_full = f'Density - {usecase_name} - full - nonzero samples'
        plt_dist.plot_density(data[feats_for_density_full], density_dir_usecase, filename=fig_title_full,
                              fig_title=fig_title_full, omit_zero_samples=True, store_tikz=True)


    #%% Skew and kurtosis
    density_dir = "./Figures/Density"
    os.makedirs(density_dir, exist_ok=True)

    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        density_dir_usecase = os.path.join(density_dir, usecase_name)
        os.makedirs(density_dir_usecase, exist_ok=True)
        # Get data and feature set
        data = get_data(os.path.join(data_dir, dict_usecase['dataset']))
        data = feat_utils.add_features_to_data(data, dict_usecase)
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)
        data = data.astype('float')
        # Data preprocessing
        # data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], do_smoothe=True)

        feats_for_density = [feature.name for feature in feature_set.get_input_feats() if
                             not feature.cyclic and not feature.statistical]
        if usecase_name in ['CPS-Data', 'SensorA6', 'SensorB2', 'SensorC6']:
            feats_for_density.remove("holiday_weekend")
            feats_for_density.remove('daylight')
        if usecase_name == 'Solarhouse1':
            feats_for_density.remove('VDSolar_inv')
        if usecase_name == 'Solarhouse1':
            data['SGlobal'][data['SGlobal'] < 30] = 0
        if usecase_name == 'Solarhouse2':
            feats_for_density.remove('Vd_Solar_inv')

        feats_for_density_full = feats_for_density + feature_set.get_output_feature_names()

        scaler = Normalizer()
        scaler.fit(data)
        data = scaler.transform(data)

        df_skew = data_analysis.calc_skew_kurtosis(data[feats_for_density_full])
        df_skew_nonzero = data_analysis.calc_skew_kurtosis(data[feats_for_density_full], True)
        df_skew.to_csv(os.path.join(density_dir_usecase, f"{usecase_name}_skew_kurtosis.csv"), index_label='Metric')
        df_skew_nonzero.to_csv(os.path.join(density_dir_usecase, f"{usecase_name}_skew_kurtosis_nonzero.csv"), index_label='Metric')


    #%%
    ##### Stationarity/Normal Distribution
    stationarity_dir = "./Figures/Stationarity"
    os.makedirs(stationarity_dir, exist_ok=True)
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        # Get data and feature set
        data = get_data(os.path.join(data_dir, dict_usecase['dataset']))
        data = feat_utils.add_features_to_data(data, dict_usecase)
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)
        data = data.astype('float')

        features_for_corrmatrix = [feature.name for feature in feature_set.get_input_feats() if
                                   not feature.cyclic and not feature.statistical]
        if usecase_name in ['CPS-Data', 'SensorA6', 'SensorB2', 'SensorC6']:
            features_for_corrmatrix.remove("holiday_weekend")
            features_for_corrmatrix.append("holiday")

        cur_data = data[features_for_corrmatrix]
        cur_data = cur_data.dropna(axis=0)
        if usecase_name == "Solarhouse1":
            cur_data['SGlobal'][cur_data['SGlobal'] < 30] = 0

        cur_data = cur_data.dropna(axis=0)
        expanded_features = train_utils.expand_features(cur_data, features_for_corrmatrix, expander_parameters)

        if usecase_name == 'Solarhouse2' or usecase_name == 'Solarhouse1':
            df_tests = data_analysis.norm_stat_tests(cur_data)
            df_tests.to_csv(os.path.join(stationarity_dir, f'Tests_{usecase_name}.csv'), index_label='Feature')
            df_tests_exp = data_analysis.norm_stat_tests(expanded_features)
            df_tests_exp.to_csv(os.path.join(stationarity_dir, f'Tests_{usecase_name}_exp.csv'), index_label='Feature')
