from ModelTraining.feature_engineering.experimental import DynamicFeaturesSampleCut
import numpy as np
import pandas as pd


def test_lookback_dynfeats_samplecut():
    vals = np.random.randn(100,4)
    lookback = 5
    feat_names = ['feat_1', 'feat_2', 'feat_3', 'target_feat']
    data = pd.DataFrame(data=vals,columns=feat_names)
    lookback_tr = DynamicFeaturesSampleCut(features_to_transform=[True, True, False], lookback_horizon=lookback,
                                             flatten_dynamic_feats=True)
    data_tr, target_tr = lookback_tr.fit_resample(data[['feat_1','feat_2','feat_3']], data['target_feat'])
    assert(np.all(target_tr == data['target_feat'][lookback:]))
    for feat in range(1,3):
        for i in range(1, lookback):
            assert(np.all(data_tr[f'feat_{feat}_{i}'].values == data[f'feat_{feat}'].values[lookback-i:-i]))


if __name__ == "__main__":
    test_lookback_dynfeats_samplecut()
