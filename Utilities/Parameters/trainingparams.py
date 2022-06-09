from . parameters import Parameters
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingParams(Parameters):
    model_type: str = "LinearRegression"
    model_name: str = ""
    static_input_features: List[str] = None
    dynamic_input_features: List[str] = None
    target_features: List[str] = None
    lookback_horizon: int = 0
    prediction_horizon: int = 1
    training_split: float = 0.7
    flatten_dynamic_input_features: bool = True
    normalizer: str = "IdentityScaler"
    expansion: List[str] = field(default_factory=list)
    epochs: int = 0
    dynamic_output_features: List[str] = field(default_factory=list)

    def str_target_feats(self):
        return "_".join(self.target_features)
