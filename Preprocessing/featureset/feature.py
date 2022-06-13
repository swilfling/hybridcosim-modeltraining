from dataclasses import dataclass
from typing import List


@dataclass
class Feature:
    name: str = ""
    models: List[str] = None
    datatype: str = ""
    static: bool = False
    input: bool = False
    output: bool = False
    parameter: bool = False
    dynamic: bool = False
    cyclic: bool = False
    statistical: bool = False
    init: float = None
    description: str = ""