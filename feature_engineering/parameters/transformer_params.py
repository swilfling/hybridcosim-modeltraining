from dataclasses import dataclass
from .parameters import Parameters
from dataclasses import field

@dataclass
class TransformerParams(Parameters):
    type: str = ""
    params: dict = field(default_factory=dict)