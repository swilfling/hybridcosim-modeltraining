import json
from . import BasicInterface


class JSONInterface(BasicInterface):
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
        dict_file = {"Type": type(self).__name__, "Parameters": self.__dict__}
        return str(json.dumps(dict_file))

    @classmethod
    def from_json(cls, dict_file):
        for subclass in cls._get_subclasses():
            if dict_file["Type"] in str(subclass):
                return subclass(**(dict_file["Parameters"]))
        return cls(**(dict_file["Parameters"]))

    @classmethod
    def load(cls, filename="testbench_params.json"):
        with open(filename, "r") as f:
            return cls.from_json(json.load(f))