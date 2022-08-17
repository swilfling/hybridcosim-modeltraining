from ModelTraining.Utilities.MetricsExport import MetricsVal, MetricsCalc
from ModelTraining.Utilities.trainingdata import TrainingData
import numpy as np


def test_metrics_calc():
    metrics_calc = MetricsCalc(metr_names={'Metrics': ['rsquared', 'cvrmse']})

    y_true = np.ones((100, 2))
    y_pred = np.zeros((100, 2))

    result = TrainingData(test_target=y_true, test_prediction=y_pred, target_feat_names=['Feat1', 'Feat2'])

    df_metrs = metrics_calc.calc_perf_metrics_df(result, df_index=["TestMetr"])
    print(df_metrs)
    assert(df_metrs['rsquared_Feat1']["TestMetr"] == -99)
    assert (df_metrs['rsquared_Feat2']["TestMetr"] == -99)


if __name__ == "__main__":
    test_metrics_calc()
