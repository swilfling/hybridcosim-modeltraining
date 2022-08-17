from ModelTraining.Data.DataImport.dataimport import DataImport
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV
from imblearn.pipeline import Pipeline, make_pipeline
from ModelTraining.feature_engineering.featureengineering.resamplers import SampleCut
from ModelTraining.feature_engineering.experimental import DynamicFeaturesSampleCut
from ModelTraining.feature_engineering.featureengineering.timebasedfeatures import DynamicFeatures

class SamplingPipeline(Pipeline):
    def predict(self, X, **predict_params):
        Xt = X
        for _, name, transform in self._iter(with_final=False, filter_resample=False):
            if hasattr(transform, 'transform'):
                Xt = transform.transform(Xt)
            if hasattr(transform, 'fit_resample'):
                Xt, _ = transform.fit_resample(Xt, None)
        return self.steps[-1][1].predict(Xt, **predict_params)


if __name__ == "__main__":
    data = DataImport.load("../../Data/Configuration/DataImport/AEE/Solarhouse1/Resampled15min.json").import_data(
        "../Data/Data/AEE/Solarhouse1/Resampled15min")

    # Trial implementation
    data_feats = data[['VDSolar','TSolarRL', 'SGlobal']]
    lookback_tr = DynamicFeatures(features_to_transform=[True, True, True], lookback_horizon=5, flatten_dynamic_feats=True)
    lookback = 5

#%% Grid search:

    lookback_tr_3 = DynamicFeaturesSampleCut(lookback_horizon=lookback, flatten_dynamic_feats=True)

    #reg = SamplingPipeline(steps=[('dynamicfeaturessamplecut', lookback_tr_3), ('linearregression',LinearRegression())])
    #reg = TransformedTargetRegressor(regressor=make_pipeline(DynamicFeatures(lookback_horizon=3, flatten_dynamic_feats=True),LinearRegression()), transformer=SampleCut(lookback), check_inverse=False)

    reg = make_pipeline(DynamicFeatures(lookback_horizon=3, flatten_dynamic_feats=True), SampleCut(num_samples=3), Ridge())
 #   reg = SamplingPipeline([('dynamicfeatures',DynamicFeatures(lookback_horizon=3, flatten_dynamic_feats=True)),
 #                           ('samplecut_imblearn', SampleCut(num_samples=3)),
 #                           ('linearregression',LinearRegression())])

    x_train = data[['TSolarRL','SGlobal','VDSolar']][0:1000]
    y_train = data['TSolarVL'][0:1000]

    x, y = lookback_tr_3.fit_resample(x_train, y_train)
    print(x.shape)
    print(y.shape)

    reg.fit(x_train, y_train)

    y_test = data['TSolarVL'][100:200].to_numpy()

    search = GridSearchCV(reg, {'ridge__alpha': [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 150, 200, 500],
                                'dynamicfeaturessamplecut__lookback_horizon': [2, 3, 4, 5, 10, 15]}, verbose=4)

    search = GridSearchCV(reg,
                          {'ridge__alpha': [0.1],
                          #'dynamicfeaturessamplecut__lookback_horizon': [2, 3]
                          'dynamicfeatures__lookback_horizon': [2, 3],
                          'samplecut__num_samples': [2, 3]
                        },
                          scoring=['r2','neg_mean_squared_error'],
                          refit='r2',
                          verbose=4)
    search.fit(x_train, y_train)
    print(search.best_params_)
    reg.set_params(**search.best_params_)
    reg.fit(x_train, y_train)

#%%
    x_pr = reg.predict(data[['TSolarRL', 'SGlobal', 'VDSolar']][100:200])
    print(x_pr.shape)
    #print(data.shape[0])
    plt.figure()
    plt.plot(y_test[2:])
    plt.plot(x_pr)
    plt.show()



#%%
    from sklearn.compose import TransformedTargetRegressor
    lookback_tr_3 = DynamicFeatures(lookback_horizon=lookback, flatten_dynamic_feats=True)
    reg_2 = make_pipeline(lookback_tr_3, Ridge())
    reg = TransformedTargetRegressor(regressor=reg_2, transformer=SampleCut(lookback), check_inverse=False)

    x_train = data[['TSolarRL','SGlobal','VDSolar']][0:1000]
    y_train = data['TSolarVL'][0:1000]
    reg.fit(x_train, y_train)
    y_test = data['TSolarVL'][100:200].to_numpy()

    search = GridSearchCV(reg, {'regressor__dynamicfeatures__lookback_horizon': [2, 3, 4, 5, 10, 15],
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