from ModelTraining.feature_engineering.transformers import Boxcox
from ModelTraining.datamodels.datamodels.processing import Normalizer
from ModelTraining.dataimport import DataImport
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = DataImport.load("../Configuration/DataImport/Resampled15min.json").import_data("../../Data/AEE/Resampled15min")
    boxcox = Boxcox(omit_zero_samples=True, features_to_transform=['VDSolar'])
    data_tr = boxcox.transform(Normalizer().fit(data).transform(data))

    plt.plot(data['VDSolar'][0:1000])
    plt.plot(data_tr['VDSolar'][0:1000])
    plt.show()
