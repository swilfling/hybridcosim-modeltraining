from . import TrainingParams
from .transformer_params import TransformerParams
from dataclasses import dataclass
from typing import List


@dataclass
class TrainingParamsExpanded(TrainingParams):
    """
    Training parameters for expanded model
    """
    transformer_params: List[TransformerParams] = None

    def str_expansion(self):
        """
        String identifier for expansion
        """
        return "_".join(tr.type for tr in self.transformer_params) if self.transformer_params is not None else ""

    def _get_attrs(self):
        attrs = self.__dict__.copy()
        attrs.update({'transformer_params': [params._get_attrs() for params in self.transformer_params]})
        return attrs

    def _set_attrs(self, **kwargs):
        transformers = kwargs.pop('transformer_params', None)
        super()._set_attrs(**kwargs)
        if transformers is not None:
            self.transformer_params = [TransformerParams(**params) for params in transformers]