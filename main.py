import os
from Utilities.Parameters import TrainingResults, TrainingParams
import ModelTrainingUtilities.training_utils as train_utils
import Utilities.Plotting.plotting_utilities as plt_utils
import datamodels.datamodels.validation.metrics as metrics
from Utilities.feature_set import FeatureSet
import pandas as pd
from run_training_and_test import run_training_and_test

def parse_excel(file, index_col="datetime"):
    df = pd.read_excel(file)
    df = df.rename({'time':'daytime'},axis=1)
    index = pd.Index(pd.to_datetime(df[index_col]))
    df = df.set_index(index)
    df = df.drop([index_col], axis=1)
    df = df.rename({df.columns[0]: 'energy'}, axis=1)
    return df


if __name__ == '__main__':
    path = "data/data.xlsx"
    data = parse_excel(path)

    # Added: Preprocessing - Smooth features
    #smoothe_data = True
    print("Starting Training")
    # Training parameters
    model_type = "LinearRegression"
    normalizer = "IdentityScaler"
    train_frac = 0.8
    prediction_horizon = 1
    lookback_horizon = 4

    expansion = ["IdentityExpander"]

    trainparams_basic = TrainingParams(model_type=model_type,
                                       lookback_horizon=lookback_horizon,
                                       prediction_horizon=prediction_horizon,
                                       training_split=train_frac,
                                       normalizer=normalizer,
                                       expansion=expansion)
    results_path = "./results"
    feature_set = FeatureSet('Configuration/FeatureSet/FeatureSet_cps_data.csv')
    list_training_parameters = [train_utils.set_train_params_model(trainparams_basic, feature_set, feature, model_type)
                                for feature in feature_set.get_output_feature_names() ]
    models, results = run_training_and_test(data, list_training_parameters, results_path, do_predict=True)
    # Save Model
    import numpy as np
    metrs = [metrics.all_metrics(y_true=result.test_target, y_pred=np.array(result.test_prediction)) for result in results]
    print(metrs)
    for result in results:
        plt_utils.scatterplot(np.array(result.test_prediction), result.test_target)

    if model_type == 'SymbolicRegression':
        print(models[0].model._program)
