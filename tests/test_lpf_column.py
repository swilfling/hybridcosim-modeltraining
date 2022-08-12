from ModelTraining.Data.DataImport.dataimport import DataImport
from ModelTraining.feature_engineering.featureengineering.filters import ButterworthFilter, Filter
from ModelTraining.feature_engineering.featureengineering.compositetransformers import Transformer_MaskFeats
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def test_filter_params_baseclass():
    filter_1 = ButterworthFilter(T=20, order=3)
    params_but = filter_1.get_params(deep=True)
    filter_basic = Filter()
    params_basic = filter_basic.get_params(deep=True)
    # Are params different from each other?
    assert(params_but != params_basic)
    # Are basic params in subclass params?
    assert([param in params_but for param in params_basic])


def test_store_load_filter():
    path = "./test_output"
    filename = "LPF.pickle"

    filter_1 = ButterworthFilter(T=20, order=3)
    filter_1.save_pkl(path, filename)
    filter_2 = ButterworthFilter.load_pkl(path, filename)
    assert(filter_1.get_params() == filter_2.get_params())
    assert(filter_1.get_params(deep=True) == filter_2.get_params(deep=True))


def test_store_load_filter_baseclass():
    path = "./test_output"
    filename = "LPF.pickle"

    filter_1 = ButterworthFilter(T=20, order=3)
    filter_1.save_pkl(path, filename)

    filter_2 = ButterworthFilter.load_pkl(path, filename)
    filter_3 = Filter.load_pkl(path, filename)
    assert(filter_2.get_params() == filter_3.get_params())
    assert (filter_2.get_params(deep=True) == filter_3.get_params(deep=True))


def test_columntransformer():


    data = DataImport.load("../Configuration/DataImport/Resampled15min.json").import_data(
        "../../Data/AEE/Resampled15min")
    data = data[['TSolarVL', 'TSolarRL', 'VDSolar', 'SGlobal']]
    features_to_smoothe = ['TSolarVL']
    filter_mask = Transformer_MaskFeats(features_to_transform=[True, False, False, False],
                                        transformer_type='ButterworthFilter', transformer_params={'T': 20})
    transf = ColumnTransformer(
        [("filter", ButterworthFilter(T=20), [feat in features_to_smoothe for feat in data.columns])],
        remainder='passthrough').fit(data)
    data_tr_1 = pd.DataFrame(index=data.index, columns=data.columns, data=transf.transform(data))
    data_tr_2 = filter_mask.fit_transform(data)
    assert(np.all(data_tr_1 - data_tr_2 == 0))


if __name__ == "__main__":

    test_filter_params_baseclass()
    test_store_load_filter()
    test_store_load_filter_baseclass()

    data = DataImport.load("../Data/Configuration/DataImport/AEE/Solarhouse1/Resampled15min.json").import_data(
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

    test_columntransformer()