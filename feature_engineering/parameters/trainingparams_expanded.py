from . import TrainingParams
from dataclasses import dataclass
from typing import List


@dataclass
class TrainingParamsExpanded(TrainingParams):
    """
    Training parameters for expanded model
    """
    transformers: List[dict] = None

    def str_expansion(self):
        """
        String identifier for expansion
        """
        return "_".join(tr['Type'] for tr in self.transformers) if self.transformers is not None else ""

    