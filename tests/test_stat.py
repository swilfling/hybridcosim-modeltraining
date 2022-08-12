from ModelTraining.Data.DataImport.dataimport import DataImport
from ModelTraining.feature_engineering.experimental import StatisticalFeatures
from ModelTraining.feature_engineering.featureengineering.timebasedfeatures import StatisticalFeaturesNumpy
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    data = DataImport.load("../Data/Configuration/DataImport/AEE/Solarhouse1/Resampled15min.json").import_data(
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