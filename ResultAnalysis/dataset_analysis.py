#%%

import ModelTraining.Preprocessing.FeatureCreation.add_features as feat_utils
import ModelTraining.Preprocessing.get_data_and_feature_set
from ModelTraining.Preprocessing.data_analysis import calc_vif_df
from ModelTraining.Utilities.Parameters import TrainingParams
import ModelTraining.Training.TrainingUtilities.training_utils as train_utils
import ModelTraining.Preprocessing.DataPreprocessing.data_preprocessing as dp_utils
import ModelTraining.Utilities.MetricsExport.export_metrics as metr_exp
import ModelTraining.Preprocessing.DataImport.data_import as data_import
import os


if __name__ == '__main__':
    root_dir = "../"
    data_dir = "../../"
    # Added: Preprocessing - Smooth features
    usecase_config_path = os.path.join(root_dir, 'Configuration/UseCaseConfig')
    list_usecases = ['CPS-Data', 'Solarhouse1','Solarhouse2']

    dict_usecases = [data_import.load_from_json(os.path.join(usecase_config_path, f"{name}.json")) for name in
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
        data, feature_set = ModelTraining.Preprocessing.get_data_and_feature_set.get_data_and_feature_set(os.path.join(data_dir, dict_usecase['dataset']),
                                                                                                          os.path.join(root_dir, dict_usecase['fmu_interface']))
        # Add features to dataset
        data, feature_set = feat_utils.add_features(data, feature_set, dict_usecase)
        # Data preprocessing
        data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], do_smoothe=True)
        # Export correlation matrices
        metr_exp.export_corrmatrices(data[feature_set.get_input_feature_names()], matrix_path,
                                     usecase_name, True, expander_parameters=expander_parameters)


#%% VIF calculation
    for dict_usecase in dict_usecases:
        usecase_name = dict_usecase['name']
        # Get data and feature set
        data, feature_set = ModelTraining.Preprocessing.get_data_and_feature_set.get_data_and_feature_set(os.path.join(data_dir, dict_usecase['dataset']),
                                                                                                          os.path.join(root_dir, dict_usecase['fmu_interface']))
        data, feature_set = feat_utils.add_features(data, feature_set, dict_usecase)
        data = data.astype('float')
        # Data preprocessing
        data = dp_utils.preprocess_data(data, dict_usecase['to_smoothe'], do_smoothe=True)
        training_params = TrainingParams(training_split=0.8,static_input_features=feature_set.get_static_feature_names(),target_features=feature_set.get_output_feature_names(),
                                         dynamic_input_features=[])
        static_data = data[training_params.static_input_features]
        vif = calc_vif_df(static_data.values, static_data.columns, dropinf=False)
        vif.to_csv(f'{vif_path}/vif_{usecase_name}_full.csv', float_format=float_format, index_label='Feature')
        vif = calc_vif_df(static_data.values, static_data.columns, dropinf=True)
        vif.to_csv(f'{vif_path}/vif_{usecase_name}.csv',float_format=float_format,index_label='Feature')
        print(vif)
        expanded_features = train_utils.expand_features(static_data, training_params.static_input_features, [],expander_parameters=expander_parameters)
        vif_expanded = calc_vif_df(expanded_features.values, expanded_features.columns, True)
        print(vif_expanded)
        vif_expanded.to_csv(f'{vif_path}/vif_expanded_{usecase_name}_full.csv',float_format=float_format, index_label='Feature')
