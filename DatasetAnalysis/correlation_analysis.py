#%%

import ModelTraining.Preprocessing.add_features as feat_utils
from ModelTraining.feature_engineering.featureset import FeatureSet
from ModelTraining.dataimport import DataImport
import ModelTraining.Preprocessing.data_analysis as data_analysis
import ModelTraining.Training.TrainingUtilities.training_utils as train_utils
import ModelTraining.Preprocessing.data_preprocessing as dp_utils
import ModelTraining.dataimport.data_import as data_import
import ModelTraining.Utilities.Plotting.plot_distributions as plt_dist
import os
from ModelTraining.datamodels.datamodels.processing.datascaler import Normalizer


if __name__ == '__main__':
    #%%
    root_dir = "../"
    data_dir = "../../"
    # Added: Preprocessing - Smooth features
    config_path = os.path.join(root_dir, 'Configuration')
    list_usecases = ['CPS-Data', 'SensorA6', 'SensorB2', 'SensorC6', 'Solarhouse1','Solarhouse2']

    dict_usecases = [data_import.load_from_json(os.path.join(config_path,"UseCaseConfig", f"{name}.json")) for name in
                     list_usecases]

    interaction_only=True
    matrix_path = "./Figures/Correlation"
    vif_path = './Figures/Correlation/VIF'
    os.makedirs(vif_path, exist_ok=True)
    float_format="%.2f"
    expander_parameters = {'degree': 2, 'interaction_only': True, 'include_bias': False}

    #%% correlation matrices
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        # Get data and feature set
        data_import = DataImport.load(
            os.path.join(config_path, "DataImport", f"{dict_usecase['dataset_filename']}.json"))
        data = data_import.import_data(
            os.path.join(data_dir, dict_usecase['dataset_dir'], dict_usecase['dataset_filename']))
        data = feat_utils.add_features_to_data(data, dict_usecase)
        feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
        feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)
        # Data preprocessing
        data = dp_utils.preprocess_data(data,dict_usecase['dataset_filename'])
        # Export correlation matrices
        features_for_corrmatrix = [feature.name for feature in feature_set.get_input_feats() if not feature.cyclic and not feature.statistical]

        if data.shape[1] > 1:
            filename_basic = f'Correlation_{usecase_name}_IdentityExpander'
            corr = data_analysis.corrmatrix(data[features_for_corrmatrix])
            plt_dist.printHeatMap(corr, matrix_path, filename_basic, vmin=-1, vmax=1, cmap='coolwarm', annot=True, fmt='.2f')
            data_analysis.reshape_corrmatrix(corr).to_csv(os.path.join(matrix_path, f'{filename_basic}_flat.csv'))

            filename_exp = f'Correlation_{usecase_name}_PolynomialExpansion'
            expanded_features = train_utils.expand_features(data, features_for_corrmatrix, expander_parameters=expander_parameters)
            corr_exp = data_analysis.corrmatrix(expanded_features)
            data_analysis.reshape_corrmatrix(corr).to_csv(os.path.join(matrix_path, f'{filename_exp}_flat.csv'))
            plt_dist.printHeatMap(corr_exp, matrix_path, filename_exp, vmin=-1, vmax=1, cmap='coolwarm', annot=True, fmt='.2f')


#%% VIF calculation
    for dict_usecase in dict_usecases:
         usecase_name = dict_usecase['name']
         # Get data and feature set
         data_import = DataImport.load(
             os.path.join(config_path, "DataImport", f"{dict_usecase['dataset_filename']}.json"))
         data = data_import.import_data(
             os.path.join(data_dir, dict_usecase['dataset_dir'], dict_usecase['dataset_filename']))
         data = feat_utils.add_features_to_data(data, dict_usecase)
         feature_set = FeatureSet(os.path.join(root_dir, dict_usecase['fmu_interface']))
         feature_set = feat_utils.add_features_to_featureset(feature_set, dict_usecase)

         data = data.astype('float')
         # Data preprocessing
         data = dp_utils.preprocess_data(data, dict_usecase['dataset_filename'])

         features_for_corrmatrix = [feature.name for feature in feature_set.get_input_feats() if
                                    not feature.cyclic and not feature.statistical]

         static_data = data[features_for_corrmatrix]
         #static_data_norm = (static_data - np.nanmean(static_data, axis=0)) / np.nanstd(static_data, axis=0)
         scaler = Normalizer()
         scaler.fit(static_data)
         static_data_norm = scaler.transform(static_data)

         vif_norm = data_analysis.calc_vif_df(static_data_norm, dropinf=False)
         vif_norm = vif_norm.rename({"VIF":"VIF normalized"}, axis=1)
         vif_full = data_analysis.calc_vif_df(static_data, dropinf=False)
         vif_full.to_csv(f'{vif_path}/vif_{usecase_name}_full.csv', float_format=float_format, index_label='Feature')
         vif = data_analysis.calc_vif_df(static_data, dropinf=True)
         vif.to_csv(f'{vif_path}/vif_{usecase_name}.csv',float_format=float_format,index_label='Feature')
         print(vif_full)
         print(vif_norm)

         expanded_features = train_utils.expand_features(data, static_data.columns,expander_parameters=expander_parameters)
         vif_expanded = data_analysis.calc_vif_df(expanded_features, True)

         scaler.fit(expanded_features)
         expanded_features_norm = scaler.transform(expanded_features)
         vif_norm = data_analysis.calc_vif_df(expanded_features_norm, dropinf=False)
         vif_expanded_norm = vif_norm.rename({"VIF": "VIF normalized"}, axis=1)

         print(vif_expanded)
         print(vif_expanded_norm)
         vif_expanded.to_csv(f'{vif_path}/vif_expanded_{usecase_name}_full.csv',float_format=float_format, index_label='Feature')

