from ModelTraining.feature_engineering.featureengineering.filters import ButterworthFilter, Filter
from ModelTraining.feature_engineering.featureengineering.compositetransformers import Transformer_MaskFeats
from sklearn.compose import ColumnTransformer
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
    data = pd.DataFrame(data=np.random.randn(100,4), columns=[f'feat_{i}' for i in range(4)])
    filter_mask = Transformer_MaskFeats(mask_type='MaskFeats_Inplace',
        mask_params={'features_to_transform':[True, False, False, False],
                     'rename_df_cols':False},
                    transformer_type='ButterworthFilter', transformer_params={'T': 20})
    transf = ColumnTransformer(
        [("filter", ButterworthFilter(T=20), [True, False, False, False])], remainder='passthrough').fit(data)
    data_tr_1 = pd.DataFrame(index=data.index, columns=data.columns, data=transf.transform(data))
    data_tr_2 = filter_mask.fit_transform(data)
    assert(np.all(data_tr_1 - data_tr_2 == 0))


if __name__ == "__main__":

    test_filter_params_baseclass()
    test_store_load_filter()
    test_store_load_filter_baseclass()
    test_columntransformer()

