from ModelTraining.feature_engineering.featureengineering.transformers import Boxcox
from ModelTraining.feature_engineering.featureengineering.compositetransformers import Transformer_MaskFeats
from ModelTraining.datamodels.datamodels.processing import Normalizer
from ModelTraining.dataimport import DataImport
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def test_transformer_maskfeats():
    data = 9 * np.ones((100, 3))
    mask_params = {'features_to_transform': [True, False, False]}
    # Transformer inplace
    tr = Transformer_MaskFeats(transformer_type="SqrtTransform", mask_type="MaskFeats_Inplace", mask_params=mask_params)
    data_tr = tr.fit_transform(data)
    # Check shape of transformed samples
    assert(data_tr.shape == data.shape)
    # Check equality of transformed and non-transformed samples -> feature mask
    assert (np.all(data_tr[:, 1] == data[:, 1]))
    assert (np.all(data_tr[:, 2] == data[:, 2]))
    assert (np.all(data_tr[:, 0] != data[:, 0]))
    # Check value of transformed samples
    assert (np.all(data_tr[:, 0] == 3))


def test_transformer_maskfeats_df_no_rename():
    data = pd.DataFrame(data=(9 * np.ones((100, 3))), columns=['Feat1','Feat2','Feat3'])
    mask_params = {'features_to_transform': [True, False, False], 'rename_df_cols':False}
    # Transformer inplace
    tr = Transformer_MaskFeats(transformer_type="SqrtTransform", mask_type="MaskFeats_Inplace", mask_params=mask_params)
    data_tr = tr.fit_transform(data)
    # Check shape of transformed samples
    assert(data_tr.shape == data.shape)
    # Check equality of transformed and non-transformed samples -> feature mask
    assert (np.all(data_tr['Feat1'] != data['Feat1']))
    assert (np.all(data_tr['Feat2'] == data['Feat2']))
    assert (np.all(data_tr['Feat3'] == data['Feat3']))
    # Check value of transformed samples
    assert (np.all(data_tr['Feat1'] == 3))


def test_transformer_maskfeats_df_rename():
    data = pd.DataFrame(data=(9 * np.ones((100, 3))), columns=['Feat1','Feat2','Feat3'])
    mask_params = {'features_to_transform': [True, False, False], 'rename_df_cols':True}
    # Transformer inplace
    tr = Transformer_MaskFeats(transformer_type="SqrtTransform", mask_type="MaskFeats_Inplace", mask_params=mask_params)
    data_tr = tr.fit_transform(data)
    # Check shape of transformed samples
    assert(data_tr.shape == data.shape)

    # Check equality of transformed and non-transformed samples -> feature mask
    assert (np.all(data_tr['Feat1_sqrt'] != data['Feat1']))
    assert (np.all(data_tr['Feat2'] == data['Feat2']))
    assert (np.all(data_tr['Feat3'] == data['Feat3']))
    # Check value of transformed samples
    assert (np.all(data_tr['Feat1_sqrt'] == 3))


if __name__ == "__main__":
    test_transformer_maskfeats()
    test_transformer_maskfeats_df_rename()
    test_transformer_maskfeats_df_no_rename()

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

