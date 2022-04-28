from dataclasses import dataclass
from ModelTraining.Utilities.Parameters import Parameters
from ModelTraining.datamodels.datamodels.processing.feature_extension import StoreInterface
import numpy as np


@dataclass
class TrainingResults(Parameters, StoreInterface):
    train_index: np.ndarray = None
    train_target: np.ndarray = None
    train_prediction: np.ndarray = None
    test_index: np.ndarray = None
    test_target: np.ndarray = None
    test_prediction: np.ndarray = None
    test_input: np.ndarray = None
