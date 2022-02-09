import dataclasses
import json
from dataclasses import dataclass

@dataclass
class Parameters:
    def __init__(self, **kwargs):
        pass

    def to_file(self, file_path, append=False):
        file_path = str(file_path)
        if not file_path.endswith('.json'):
            file_path = f"{file_path}.json"
        flag = "a" if append else "w"
        with open(file_path, flag) as fp:
            fp.write(self.to_json())

    def to_json(self):
        dict_file = {"Type": type(self).__name__, "Parameters": dataclasses.asdict(self)}
        return str(json.dumps(dict_file))

    @classmethod
    def from_json(cls, dict_file):
        for subclass in cls._get_subclasses():
            if dict_file["Type"] in str(subclass):
                return subclass(**(dict_file["Parameters"]))
        return cls(**(dict_file["Parameters"]))

    @classmethod
    def _get_subclasses(cls, list_subclasses=[]):
        for subclass in cls.__subclasses__():
            subclass._get_subclasses(list_subclasses)
            list_subclasses.append(subclass)
        return list_subclasses

    @classmethod
    def load(cls, filename="testbench_params.json"):
        with open(filename, "r") as f:
            return cls.from_json(json.load(f))

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

