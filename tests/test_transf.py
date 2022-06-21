from ModelTraining.Preprocessing.DataPreprocessing.transformers import Boxcox
from ModelTraining.datamodels.datamodels.processing import Normalizer
from ModelTraining.Preprocessing.get_data_and_feature_set import get_data
import matplotlib.pyplot as plt

if __name__ == "__main__":

    data = get_data("../../Data/AEE/Resampled15min.csv")
    boxcox = Boxcox(omit_zero_samples=True, features_to_transform=['VDSolar'])
    data_tr = boxcox.transform(Normalizer().fit(data).transform(data))

    plt.plot(data['VDSolar'][0:1000])
    plt.plot(data_tr['VDSolar'][0:1000])
    plt.show()
