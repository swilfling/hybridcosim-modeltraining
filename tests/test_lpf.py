from ModelTraining.dataimport import DataImport
from ModelTraining.feature_engineering.filters import ButterworthFilter, Filter
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    filter_1 = ButterworthFilter(T=20, order=3)
    path = "../results"
    filename = "LPF.pickle"
    filter_1.save_pkl(path, filename)
    params = filter_1.get_params(deep=True)
    print(params)
    filter_basic = Filter()
    print(filter_basic.get_params(deep=True))

    filter_2 = ButterworthFilter.load_pkl(path, filename)
    print(filter_2.T)
    print(filter_2.order)
    print(filter_2.coef_)

    filter_3 = Filter.load_pkl(path, filename)
    print(filter_3.T)
    print(filter_3.order)
    print(filter_3.coef_)

    data = DataImport.load("../Configuration/DataImport/Resampled15min.json").import_data(
        "../../Data/AEE/Resampled15min")
    filter_4 = ButterworthFilter(features_to_transform=[True, False, False, False], T=20)
    filter_4.fit(data[['TSolarVL', 'TSolarRL', 'VDSolar', 'SGlobal']])
    data_tr = filter_4.transform(data[['TSolarVL', 'TSolarRL', 'VDSolar', 'SGlobal']])

    plt.plot(data['TSolarVL'][0:200])
    plt.plot(data_tr['TSolarVL'][0:200])
    plt.plot(data['TSolarRL'][0:200])
    plt.plot(data_tr['TSolarRL'][0:200])
    plt.legend(['TSolarVL', 'TSolarVL_transf','TSolarRL', 'TSolarRL_transf'])
    plt.show()

    data = data[['TSolarVL', 'TSolarRL', 'VDSolar', 'SGlobal']]
    features_to_smoothe = ['TSolarVL']
    filter_mask = ButterworthFilter(features_to_transform=[True, False, False, False], T=20)
    transf = ColumnTransformer([("filter", ButterworthFilter(T=20), [feat in features_to_smoothe for feat in data.columns])],
                               remainder='passthrough').fit(data)
    data_tr_1 = pd.DataFrame(index=data.index, columns=data.columns, data=transf.transform(data))
    data_tr_2 = filter_mask.fit_transform(data)
    import numpy as np
    print(np.all(data_tr_1 - data_tr_2 == 0))