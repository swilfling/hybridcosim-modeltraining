from ModelTraining.feature_engineering.featurecreators.dynamicfeatures import DynamicFeatures, SampleCut
from ModelTraining.Training.TrainingUtilities.training_utils import extract_training_and_test_set
from ModelTraining.dataimport import DataImport
from ModelTraining.feature_engineering.parameters import TrainingParams
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

if __name__ == "__main__":
    data = DataImport.load("../Configuration/DataImport/Resampled15min.json").import_data("../../Data/AEE/Resampled15min")

    # Trial implementation
    data_feats = data[['VDSolar','TSolarRL', 'SGlobal']]
    lookback_tr = DynamicFeatures(features_to_transform=[True, True, True], lookback_horizon=5, flatten_dynamic_feats=True)

    lookback_tr.fit(data_feats)
    data_tr = lookback_tr.transform(data_feats)
    feat_names = lookback_tr.get_feature_names_out(['VDSolar','TSolarRL', 'SGlobal'])
    print(feat_names)

    # Test 1:
    lookback = 5
    lookback_tr_2 = DynamicFeatures(features_to_transform=[True, True, False], lookback_horizon=lookback, flatten_dynamic_feats=True)

    lookback_tr_2.fit(data_feats)
    data_tr_2 = lookback_tr_2.transform(data_feats)
#    feat_names_2 = lookback_tr_2.get_feature_names_out(['VDSolar','TSolarRL', 'SGlobal'])
#    print(feat_names_2)

    # Test: non-transformed data should stay the same except for first samples
    plt.plot(data['SGlobal'][0:100])
    plt.plot(data_tr_2['SGlobal'][0:100])
    assert(np.all(data['SGlobal'][lookback:] - data_tr_2['SGlobal'] == 0))
    plt.show()

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
        x1 = data_tr_2[feat_name][lookback:].to_numpy().flatten()
        x2 = x[...,i].flatten()
        plt.plot(x1[70:100], marker='x')
        plt.plot(x2[70:100])
        plt.legend([feat_name,'extracted'])
        plt.show()
        #assert(np.all((x1 - x2)==0))

#%%
    lookback_tr_3 = DynamicFeatures(lookback_horizon=lookback, flatten_dynamic_feats=True)
    reg_2 = make_pipeline(lookback_tr_3, Ridge())
    reg = TransformedTargetRegressor(regressor=reg_2, transformer=SampleCut(lookback), check_inverse=False)

    x_train = data[['TSolarRL','SGlobal','VDSolar']][0:1000]
    y_train = data['TSolarVL'][0:1000]
    reg.fit(x_train, y_train)
    y_test = data['TSolarVL'][100:200].to_numpy()

    search = GridSearchCV(reg, {'regressor__ridge__alpha': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 150, 200, 500],
                                'regressor__dynamicfeatures__lookback_horizon': [2, 3, 4, 5, 10, 15],
                                'transformer__num_samples': [2, 3, 4, 5, 10, 15]}, scoring='r2')
    search.fit(x_train,y_train)
    print(search.best_params_)
    reg.set_params(**search.best_params_)
    x_pr = reg.predict(data[['TSolarRL', 'SGlobal', 'VDSolar']][100:200])
    print(x_pr.shape)
    #print(data.shape[0])
    plt.figure()
    plt.plot(y_test)
    plt.plot(x_pr)
    plt.show()