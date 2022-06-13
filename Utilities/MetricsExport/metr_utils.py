import json
from datetime import datetime


def create_file_name_timestamp():
    """
    Create string containing current timestamp
    """
    return "Experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def dict_to_json(dict_vals: dict = {}, filename: str = "./metrics.json"):
    """
    Export dictionary to json file
    @param dict_vals: dictionarya to store
    @param filename: full path to file
    """
    with open(filename, 'w') as f:
        json.dump(dict_vals, f)


