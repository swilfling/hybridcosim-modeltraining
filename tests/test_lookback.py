from ModelTraining.dataimport import DataImport
import matplotlib.pyplot as plt
from ModelTraining.feature_engineering.experimental import DynamicFeaturesSampleCut
from ModelTraining.feature_engineering.timebasedfeatures import DynamicFeatures
import numpy as np
from ModelTraining.feature_engineering.parameters import TrainingParams
from ModelTraining.Training.TrainingUtilities.training_utils import extract_training_and_test_set


if __name__ == "__main__":
    data = DataImport.load("../Configuration/DataImport/Resampled15min.json").import_data("../../Data/AEE/Resampled15min")

    # Trial implementation
    data_feats = data[['VDSolar','TSolarRL', 'SGlobal']]
    lookback_tr = DynamicFeatures(lookback_horizon=5, flatten_dynamic_feats=True)

    lookback_tr.fit(data_feats)
    data_tr = lookback_tr.transform(data_feats)
    feat_names = lookback_tr.get_feature_names_out(['VDSolar','TSolarRL', 'SGlobal'])
    print(feat_names)

    # Test 1:
    lookback = 5
    lookback_tr_2 = DynamicFeaturesSampleCut(features_to_transform=[True, True, False], lookback_horizon=lookback, flatten_dynamic_feats=True)

    lookback_tr_2.fit(data_feats, data_feats['VDSolar'])
    data_tr_2, _ = lookback_tr_2.fit_resample(data_feats, data_feats['VDSolar'])
    feat_names_2 = lookback_tr_2.get_feature_names_out(np.array(['VDSolar','TSolarRL', 'SGlobal']))
    print(feat_names_2)

    # Test: non-transformed data should stay the same except for first samples
    plt.figure()
    plt.plot(data['SGlobal'][0:100])
    plt.plot(data_tr_2['SGlobal'][0:100])
    plt.show()

    assert(np.all(data['SGlobal'][lookback:] - data_tr_2['SGlobal'] == 0))

    # Plot: lookback
    plt.plot(data_tr_2['TSolarRL'][70:100])
    plt.plot(data_tr_2['TSolarRL_1'][70:100])
    plt.plot(data_tr_2['TSolarRL_2'][70:100])
    plt.plot(data_tr_2['TSolarRL_3'][70:100])
    plt.plot(data_tr_2['TSolarRL_4'][70:100])
    plt.plot(data_tr_2['TSolarRL_5'][70:100])
    plt.legend(['TSolarRL','TSolarRL_1','TSolarRL_2', 'TSolarRL_3', 'TSolarRL_4','TSolarRL_5'])
    plt.show()

    train_params = TrainingParams(lookback_horizon=lookback, dynamic_input_features=['TSolarRL','VDSolar'], static_input_features=['SGlobal'], target_features=['TSolarVL'], prediction_horizon=0)
    index, x, y, feat_names = extract_training_and_test_set(data, training_params=train_params)
    for i, feat_name in enumerate(feat_names):
        x1 = data_tr_2[feat_name].to_numpy().flatten()
        x2 = x[...,i].flatten()
        plt.plot(x1[70:100], marker='x')
        plt.plot(x2[70:100])
        plt.legend([feat_name,'extracted'])
        plt.show()
        assert(np.all((x1 - x2)==0))
