from .parameters import Parameters
from dataclasses import dataclass


@dataclass
class FeatureSelectionParams(Parameters):
    """ Feature Selection Parameters for model training """
    # Selection type
    #    Currently supported types:
    # ['R-value', 'F-value','forward_select']
    # 'AllFeatures' will return all features
    sel_type: str = 'AllFeatures'
    # Selection Threshold
    threshold: float = 0
    # Omit zero samples in threshold calculation
    omit_zero_samples: bool = True

    def get_full_name(self):
        return f'{self.sel_type}_{self.threshold}'
