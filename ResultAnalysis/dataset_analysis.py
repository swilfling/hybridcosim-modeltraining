#%%

import pandas as pd
import os
import ModelTraining.Utilities.DataImport.data_import as data_import
import statsmodels.stats.outliers_influence as stats
from ModelTraining.Utilities.Parameters.trainingparams import TrainingParams
import ModelTraining.TrainingUtilities.training_utils as train_utils
import ModelTraining.FeatureEngineering.FeatureCreation.cyclic_features as cyc_feats
import ModelTraining.FeatureEngineering.FeatureCreation.statistical_features as stat_feats

if __name__ == '__main__':
    root_dir = "../"
    data_dir = "../../"
    # Added: Preprocessing - Smooth features
    usecase_config_path = os.path.join(root_dir, 'Configuration/UseCaseConfig')
    list_usecases = ['CPS-Data', 'Solarhouse1','Solarhouse2']

    dict_usecases = [data_import.load_from_json(os.path.join(usecase_config_path, f"{name}.json")) for name in
                     list_usecases]

    interaction_only=True

    vif_path = './Figures/Correlation/VIF'
    os.makedirs(vif_path, exist_ok=True)
    float_format="%.2f"

#%% VIF calculation
    for dict_usecase in dict_usecases:
        # Get data and feature set
        data, feature_set = data_import.get_data_and_feature_set(os.path.join(data_dir, dict_usecase['dataset']),
                                                                 os.path.join(root_dir, dict_usecase['fmu_interface']))
        data = data.astype('float')
        usecase_name = dict_usecase['name']
        # Add cyclic and statistical features
        cyc_feats.add_cycl_feats(dict_usecase, feature_set)
        data, feature_set = stat_feats.add_stat_feats(data, dict_usecase,feature_set)

        training_params = TrainingParams(training_split=0.8,static_input_features=feature_set.get_static_feature_names(),target_features=feature_set.get_output_feature_names(),
                                         dynamic_input_features=[])

        vif = pd.DataFrame(index=training_params.static_input_features)
        vif['VIF'] = [stats.variance_inflation_factor(data[training_params.static_input_features].values,i) for i in range(len(training_params.static_input_features))]
        #print(vif)
        vif.to_csv(f'{vif_path}/vif_{usecase_name}_full.csv',float_format=float_format,index_label='Feature')
        pd.options.mode.use_inf_as_na = True
        vif = vif.dropna(axis=0)
        vif.to_csv(f'{vif_path}/vif_{usecase_name}.csv',float_format=float_format,index_label='Feature')
        print(vif)
        noninf_features = vif.index
        print(noninf_features)
        expanded_features = train_utils.expand_features(data, noninf_features, [],expander_parameters={"interaction_only":interaction_only})
        vif_expanded = pd.DataFrame(index=expanded_features.columns)
        vif_expanded['VIF'] = [stats.variance_inflation_factor(expanded_features.values,i) for i in range(len(expanded_features.columns))]
        vif_expanded = vif_expanded.dropna(axis=0)
        print(vif_expanded)
        vif_expanded.to_csv(f'{vif_path}/vif_expanded_{usecase_name}_full.csv',float_format=float_format, index_label='Feature')
