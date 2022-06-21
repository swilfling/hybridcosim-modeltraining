from ModelTraining.Preprocessing.DataPreprocessing.filters import ButterworthFilter, Filter
from ModelTraining.Preprocessing.get_data_and_feature_set import get_data
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filter_1 = ButterworthFilter(T=20, order=3)
    path = "../results"
    filename = "LPF.pickle"
    filter_1.save_pkl(path, filename)

    filter_2 = ButterworthFilter.load_pkl(path, filename)
    print(filter_2.T)
    print(filter_2.order)
    print(filter_2.coef_)

    filter_3 = Filter.load_pkl(path, filename)
    print(filter_3.T)
    print(filter_3.order)
    print(filter_3.coef_)

    data = get_data("../../Data/AEE/Resampled15min.csv")
    filter_4 = ButterworthFilter(features_to_transform=['TSolarVL'], T=20)
    filter_4.fit(data)
    data_tr = filter_4.transform(data)

    plt.plot(data['TSolarVL'][0:200])
    plt.plot(data_tr['TSolarVL'][0:200])
    plt.plot(data['TSolarRL'][0:200])
    plt.plot(data_tr['TSolarRL'][0:200])
    plt.legend(['TSolarVL','TSolarVL_transf','TSolarRL', 'TSolarRL_transf'])
    plt.show()