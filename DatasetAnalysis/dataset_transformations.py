#%%

import ModelTraining.Preprocessing.FeatureCreation.add_features as feat_utils
from ModelTraining.Preprocessing.get_data_and_feature_set import get_data
import ModelTraining.Preprocessing.data_analysis as data_analysis
from ModelTraining.Preprocessing.DataPreprocessing.transformers import SqrtTransform, Boxcox, Diff
import ModelTraining.Preprocessing.DataImport.data_import as data_import
import ModelTraining.Utilities.Plotting.plot_distributions as plt_dist
import ModelTraining.Utilities.Plotting.plot_data as plt_utils
from ModelTraining.Preprocessing.featureset import FeatureSet
import os
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
    dict_usecases = [dict_usecases[0]]

    interaction_only=True
    matrix_path = "./Figures/Correlation"
    vif_path = './Figures/Correlation/VIF'
    os.makedirs(vif_path, exist_ok=True)
    float_format="%.2f"
    expander_parameters = {'degree': 2, 'interaction_only': True, 'include_bias': False}


    #%%
    ##### Square root transformation
    density_dir ="./Figures/Density"
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

        feats_for_density_full = feats_for_density + feature_set.get_output_feature_names()

        data = Normalizer().fit(data).transform(data)

        output_dir = os.path.join(density_dir_usecase, 'SqrtTransformation')
        os.makedirs(output_dir, exist_ok=True)
        vals_sqrt_full = SqrtTransform(features_to_transform=feats_for_density_full).fit_transform(data)[feats_for_density_full]

        fig_title = f'Density - {usecase_name} - Square Root Transformation - nonzero samples'
        plt_dist.plot_density(vals_sqrt_full[feats_for_density], output_dir, filename=fig_title, fig_title=fig_title,
                              omit_zero_samples=True, store_tikz=False)
        fig_title_full = f'Density - {usecase_name} - Square Root Transformation - full - nonzero samples'
        plt_dist.plot_density(vals_sqrt_full, output_dir, filename=fig_title_full, fig_title=fig_title_full,
                              omit_zero_samples=True, store_tikz=False)
        for feature in feats_for_density_full:
            fig_title = f'QQ - {usecase_name} - {feature} - Sqrt - nonzero samples'
            plt_dist.plot_qq(vals_sqrt_full[feature], output_dir, filename=fig_title, fig_title=fig_title)

        df_skew = data_analysis.calc_skew_kurtosis(vals_sqrt_full)
        df_skew_nonzero = data_analysis.calc_skew_kurtosis(vals_sqrt_full, True)
        df_skew.to_csv(os.path.join(output_dir, f"{usecase_name}_sqrt_skew_kurtosis.csv"), index_label='Metric')
        df_skew_nonzero.to_csv(os.path.join(output_dir, f"{usecase_name}_sqrt_skew_kurtosis_nonzero.csv"),
                               index_label='Metric')
        df_tests = data_analysis.norm_stat_tests(vals_sqrt_full)
        df_tests.to_csv(os.path.join(output_dir, f'{usecase_name}_sqrt_tests.csv'))



    #%%
    ##### Box-cox transformation
    density_dir ="./Figures/Density"
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

        feats_for_density_full = feats_for_density + feature_set.get_output_feature_names()

        data = Normalizer().fit(data).transform(data)

        output_dir = os.path.join(density_dir_usecase, 'Box-coxTransformation')
        os.makedirs(output_dir, exist_ok=True)

        boxcox_df = Boxcox(features_to_transform=feats_for_density_full, omit_zero_samples=False).transform(data)[feats_for_density_full]

        fig_title = f'Density - {usecase_name} - Box-cox Transformation - nonzero samples'
        plt_dist.plot_density(boxcox_df[feats_for_density], output_dir, filename=fig_title, fig_title=fig_title,
                              omit_zero_samples=True, store_tikz=False)
        fig_title_full = f'Density - {usecase_name} - Box-cox Transformation - full - nonzero samples'
        plt_dist.plot_density(boxcox_df, output_dir, filename=fig_title, fig_title=fig_title,
                              omit_zero_samples=True, store_tikz=False)
        for feature in feats_for_density_full:
            fig_title = f'QQ - {usecase_name} - {feature} - Box-cox - nonzero samples'
            plt_dist.plot_qq(boxcox_df[feature], output_dir, filename=fig_title, fig_title=fig_title)

        df_skew = data_analysis.calc_skew_kurtosis(boxcox_df)
        df_skew_nonzero = data_analysis.calc_skew_kurtosis(boxcox_df, True)
        df_skew.to_csv(os.path.join(output_dir, f"{usecase_name}_boxcox_skew_kurtosis.csv"), index_label='Metric')
        df_skew_nonzero.to_csv(os.path.join(output_dir, f"{usecase_name}_boxcox_skew_kurtosis_nonzero.csv"),
                               index_label='Metric')

        df_tests = data_analysis.norm_stat_tests(boxcox_df)
        df_tests.to_csv(os.path.join(output_dir, f'{usecase_name}_boxcox_tests.csv'))




    #%%
    ##### Differencing
    density_dir ="./Figures/Density"
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

        feats_for_density_full = feats_for_density + feature_set.get_output_feature_names()

        data = Normalizer().fit(data).transform(data)
        diff_df = Diff(features_to_transform=feats_for_density_full).fit_transform(data)[feats_for_density_full]

        output_dir = os.path.join(density_dir_usecase, 'Differencing')
        os.makedirs(output_dir, exist_ok=True)

        fig_title = f'Density - {usecase_name} - Differencing - nonzero samples'
        plt_dist.plot_density(diff_df[feats_for_density], output_dir, filename=fig_title, fig_title=fig_title,
                              omit_zero_samples=True, store_tikz=False)
        fig_title_full = f'Density - {usecase_name} - Differencing - full - nonzero samples',
        plt_dist.plot_density(diff_df, output_dir, filename=fig_title_full, fig_title=fig_title_full,
                              omit_zero_samples=True, store_tikz=False)
        fig_title = f'Timeseries - Differencing'
        plt_utils.plot_data(diff_df, output_dir, fig_title=fig_title, filename=fig_title)

        for feature in feats_for_density_full:
            fig_title = f'QQ - {usecase_name} - {feature} - Differencing - nonzero samples'
            plt_dist.plot_qq(diff_df[feature], output_dir, filename=fig_title, fig_title=fig_title)

        df_skew = data_analysis.calc_skew_kurtosis(diff_df)
        df_skew_nonzero = data_analysis.calc_skew_kurtosis(diff_df, True)
        df_skew.to_csv(os.path.join(output_dir, f"{usecase_name}_diff_skew_kurtosis.csv"), index_label='Metric')
        df_skew_nonzero.to_csv(os.path.join(output_dir, f"{usecase_name}_diff_skew_kurtosis_nonzero.csv"),
                               index_label='Metric')
        diff_df = diff_df.dropna(axis=0)
        df_tests = data_analysis.norm_stat_tests(diff_df)
        df_tests.to_csv(os.path.join(output_dir, f'{usecase_name}_diff_tests.csv'))
