from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from ModelTraining.FeatureSelection.FeatureSelector import FeatureSelector
from ModelTraining.Utilities.Parameters import FeatureSelectionParams


def forward_select(x, y, training_params):

    # model = create_model(training_params, x, y, parameters=kwargs.pop('parameters',None), interaction_only=kwargs.pop('interaction_only',False))
    if x.ndim == 3:
        x = x.reshape(x.shape[0], -1)
    f_model = LinearRegression()
    # efs = EFS(f_model,
    #           min_features=2,
    #           max_features=x.shape[1],
    #           scoring='neg_mean_squared_error',
    #           cv=3)
    sfs = SFS(f_model,
              k_features='best',
              forward=True,
              n_jobs=-1)

    sfs.fit(x, y)
    return list(sfs.k_feature_idx_)


def permutation(x_train, y_train, features_names):
    if x_train.ndim == 3:
        x_train = x_train.reshape(x_train.shape[0], -1)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=False)
    
    # model = Ridge(alpha=1e-2).fit(x_train, y_train)
    model = LinearRegression().fit(x_train, y_train)
    
    model.score(x_val, y_val)
    
    scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
    
    r_multi = permutation_importance(
        model, x_val, y_val, n_repeats=30, random_state=0, scoring=scoring)
    
    for metric in r_multi:
        print(f"{metric}")
        r = r_multi[metric]
        mean_sorted = r.importances_mean.argsort()[::-1]
        for i in mean_sorted:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                print(f"    {features_names[i]:<8} "
                    f"{r.importances_mean[i]:.3f} "
                    f" +/- {r.importances_std[i]:.3f} ")
        
        fig, ax1 = plt.subplots(1, 1, figsize=(12, 8))
        ax1.barh(list(range(0, len(features_names))),mean_sorted)
        ax1.set_yticks(list(range(0, len(features_names))))
        ax1.set_yticklabels(features_names)
        
        # plt.show()


def feature_sel_model(model, x_train, y_train, feature_select_params=[FeatureSelectionParams()], training_params=None):
    selectors = []
    for expander, params in zip(model.expanders, feature_select_params):
        # Expand features and do feature selection
        expander.fit(x_train, y_train)
        selector = FeatureSelector.from_name(params.sel_type)(thresh=params.threshold,
                                                              omit_zero_samples=params.omit_zero_samples,
                                                              num_basic_features=x_train.shape[-1],
                                                              feature_names=expander.get_feature_names(
                                                                  model.feature_names))
        selector.fit_transform(expander.transform(x_train), y_train, training_params=training_params)
        expander.set_feature_select(selector.get_support())
        selector.print_metrics()
        selectors.append(selector)
        x_train = expander.fit_transform(x_train, y_train)
    return selectors