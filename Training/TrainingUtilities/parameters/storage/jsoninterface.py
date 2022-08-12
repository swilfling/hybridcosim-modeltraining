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
        dict_file = {"Type": self.__class__.__name__, "Parameters": self._get_attrs()}
        return str(json.dumps(dict_file))

    @classmethod
    def from_json(cls, dict_file):
        inst = cls.from_name(dict_file['Type'])
        if inst is not None:
            inst._set_attrs(**dict_file['Parameters'])
        return inst

    @classmethod
    def load(cls, filename="testbench_params.json"):
        with open(filename, "r") as f:
            return cls.from_json(json.load(f))