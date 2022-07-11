from ModelTraining.feature_engineering.transformers import Boxcox
from ModelTraining.feature_engineering.compositetransformers import Transformer_inplace
from ModelTraining.datamodels.datamodels.processing import Normalizer
from ModelTraining.dataimport import DataImport
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data = DataImport.load("../Configuration/DataImport/Resampled15min.json").import_data("../../Data/AEE/Resampled15min")
    data_norm = Normalizer().fit(data).transform(data)

    boxcox = Boxcox(omit_zero_samples=True)
    data_tr_1 = boxcox.transform(data_norm[['VDSolar','TSolarRL','TSolarVL']])

    # Transformer inplace
    tr = Transformer_inplace(transformer_type="SqrtTransform", features_to_transform=[True, False, False])
    data_tr_2 = tr.fit_transform(data_norm[['VDSolar','TSolarRL','TSolarVL']])

    tr = Transformer_inplace(transformer_type="Boxcox", transformer_params={'omit_zero_samples': True},features_to_transform=[True, False, False])
    data_tr_3 = tr.fit_transform(data_norm[['VDSolar','TSolarRL','TSolarVL']])

    plt.plot(data_norm['VDSolar'][0:1000])
    plt.plot(data_tr_1['VDSolar'][0:1000])
    plt.show()

    tr = Transformer_inplace(transformer_type="InverseTransform", features_to_transform=[True, True, False])
    data_tr_4 = tr.fit_transform(data[['VDSolar', 'TSolarRL', 'TSolarVL']])

    print(tr.get_feature_names_out(['VDSolar', 'TSolarRL', 'TSolarVL']))