from ModelTraining.feature_engineering.experimental import StatisticalFeatures
from ModelTraining.feature_engineering.featureengineering.timebasedfeatures import StatisticalFeaturesNumpy
from ModelTraining.feature_engineering.featureengineering.transformers import Boxcox
from ModelTraining.feature_engineering.featureengineering.compositetransformers import Transformer_MaskFeats
from ModelTraining.datamodels.datamodels.processing import Normalizer
from ModelTraining.Data.DataImport.dataimport import DataImport
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    data = DataImport.load("../../Data/Configuration/DataImport/AEE/Solarhouse1/Resampled15min.json").import_data(
        "../Data/Data/AEE/Solarhouse1/Resampled15min")
    filter_4 = Transformer_MaskFeats(features_to_transform=[True, False, False, False], transformer_type='ButterworthFilter', transformer_params={'T':20})
    filter_4.fit(data[['TSolarVL', 'TSolarRL', 'VDSolar', 'SGlobal']])
    data_tr = filter_4.transform(data[['TSolarVL', 'TSolarRL', 'VDSolar', 'SGlobal']])

    plt.plot(data['TSolarVL'][0:200])
    plt.plot(data_tr['TSolarVL'][0:200])
    plt.plot(data['TSolarRL'][0:200])
    plt.plot(data_tr['TSolarRL'][0:200])
    plt.legend(['TSolarVL', 'TSolarVL_transf','TSolarRL', 'TSolarRL_transf'])
    plt.show()

    ##################### Statistical Features #######################################################################

    data = DataImport.load("../../Data/Configuration/DataImport/AEE/Solarhouse1/Resampled15min.json").import_data(
        "../Data/Data/AEE/Solarhouse1/Resampled15min")
    stat_feats = StatisticalFeatures(window_size=24, statistical_features=['tmin','tmax'])
    import datetime
    print("Stat feats Start")
    print(datetime.datetime.now())
    data_stat = stat_feats.fit_transform(data[['TSolarRL','VDSolar']])
    print(datetime.datetime.now())
    print("Stat feats end")

    print("Stat feats NP start")
    print(datetime.datetime.now())
    stat_feats = StatisticalFeaturesNumpy(window_size=24, statistical_features=['min','max'])
    data_stat_2 = stat_feats.fit_transform(data[['TSolarRL','VDSolar']])
    print(datetime.datetime.now())
    print("Stat feats NP end")
    assert(np.all(data_stat['TSolarRL_tmax_24'].values == data_stat_2['TSolarRL_max_24'].values))
    assert (np.all(data_stat['TSolarRL_tmin_24'].values == data_stat_2['TSolarRL_min_24'].values))

    plt.figure()
    plt.plot(data['TSolarRL'][0:1000])
    plt.plot(data_stat['TSolarRL_tmin_24'][0:1000])
    plt.show()
    plt.figure()
    plt.plot(data['VDSolar'][0:1000])
    plt.plot(data_stat['VDSolar_tmin_24'][0:1000])
    plt.show()

    ############################# Transformer ####################################

    data = DataImport.load("../Configuration/DataImport/Resampled15min.json").import_data(
        "../../Data/AEE/Resampled15min")
    data_norm = Normalizer().fit(data).transform(data)
    boxcox = Boxcox(omit_zero_samples=True)
    data_tr_1 = boxcox.transform(data_norm[['VDSolar', 'TSolarRL', 'TSolarVL']])

    plt.plot(data_norm['VDSolar'][0:1000])
    plt.plot(data_tr_1['VDSolar'][0:1000])
    plt.show()
    mask_params = {'features_to_transform': [True, False, False]}
    tr = Transformer_MaskFeats(transformer_type="Boxcox", transformer_params={'omit_zero_samples': True}, mask_params=mask_params)
    data_tr_3 = tr.fit_transform(data_norm[['VDSolar', 'TSolarRL', 'TSolarVL']])

    tr = Transformer_MaskFeats(transformer_type="InverseTransform", features_to_transform=[True, True, False])
    data_tr_4 = tr.fit_transform(data[['VDSolar', 'TSolarRL', 'TSolarVL']])
    print(tr.get_feature_names_out(['VDSolar', 'TSolarRL', 'TSolarVL']))

    tr = Transformer_MaskFeats(transformer_type="PolynomialExpansion", mask_params={'features_to_transform':[True, True, False]}, mask_type='MaskFeats_Expanded')
    data_tr_4 = tr.fit_transform(data[['VDSolar', 'TSolarRL', 'TSolarVL']])
    print(tr.get_feature_names_out(['VDSolar', 'TSolarRL', 'TSolarVL']))

    tr = Transformer_MaskFeats(transformer_type="InverseTransform", features_to_transform=[True, True, False],
                               mask_type='MaskFeats_Addition')
    data_tr_4 = tr.fit_transform(data[['VDSolar', 'TSolarRL', 'TSolarVL']])
    print(tr.get_feature_names_out(['VDSolar', 'TSolarRL', 'TSolarVL']))

