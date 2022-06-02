import json
from datetime import datetime

def get_df_subset(df, label):
    cols = [col for col in df.columns if label in col]
    df_subset = df[cols]
    df_subset.columns = ["_".join(col.split("_")[:-1]) for col in cols]
    return df_subset


def create_file_name_timestamp():
    return "Experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")


def metrics_to_json(dict_metrics, filename):
    with open(filename, 'w') as f:
        json.dump(dict_metrics, f)


