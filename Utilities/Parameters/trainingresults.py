from dataclasses import dataclass
from ModelTraining.Utilities.Parameters import Parameters
import numpy as np
import pandas as pd


@dataclass
class TrainingResults(Parameters):
    train_index: pd.Series = None
    train_target: np.ndarray = None
    train_prediction: np.ndarray = None
    test_index: pd.Series = None
    test_target: np.ndarray = None
    test_prediction: np.ndarray = None
