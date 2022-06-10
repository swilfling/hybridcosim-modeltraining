import json
from datetime import datetime


def create_file_name_timestamp():
    return "Experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def metrics_to_json(dict_metrics, filename):
    with open(filename, 'w') as f:
        json.dump(dict_metrics, f)


