import json
from dataclasses import dataclass
from .storage import JSONInterface

@dataclass
class Parameters(JSONInterface):

    @staticmethod
    def store_parameters_list(parameters_list, path_full):
        with open(path_full, "w") as f:
            parameters = ",".join(params.to_json() for params in parameters_list)
            f.write("[" + parameters + "]")

    @classmethod
    def load_parameters_list(cls, path_full):
        with open(path_full, "r") as file:
            list_dicts = json.load(file)
            sim_param_list = [cls.from_json(dict) for dict in list_dicts]
            return sim_param_list

