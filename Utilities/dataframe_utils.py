import numpy as np
import pandas as pd
from pandas import DataFrame


def remove_spaces_from_labels(df):
    for label in df.columns:
        df = df.rename(columns={label: label.split(" ")[0]})
    return df


def df_set_date_index(data, column):
    data[column] = pd.to_datetime(data[column], dayfirst=True)
    return data.set_index(column)


def create_date_range(y1, m1, d1, y2, m2, d2, freq='1H'):
    return [timestamp for timestamp in
     pd.date_range(pd.Timestamp(y1, m1, d1), pd.Timestamp(y2, m2, d2), freq=freq)]


def join_list_df(list_simulation_results):
    return pd.concat(list_simulation_results)


def get_indices_from_columns(columns, features, index_offset=1):
    return [columns.get_loc(name) + index_offset for name in features]


def create_labels(list_outputs, units=[]):
    if len(units) > 0:
        return [[f"Prediction: {output} [{unit}]", f"Measurement: {output} [{unit}]"] for output, unit in zip(list_outputs, units)]
    else:
        return [[f"Prediction: {output}", f"Measurement: {output}"] for output in list_outputs]


def create_df_plotting(data: pd.DataFrame, selected_labels, label_names):
    df = data[selected_labels].copy()
    df = df.rename({label: newlabel for label, newlabel in zip(selected_labels, label_names)}, axis=1)
    return df


def create_df(trajectories, labels):
    trajectories = np.array(trajectories).T
    index = pd.TimedeltaIndex(trajectories[:,0], unit='s')
    df = DataFrame(trajectories, columns=labels, index=index)
    df = df.drop(["Time"], axis=1)
    return df


def get_df_end_values(df, labels):
    return [df[name][-1] for name in labels] if df is not None else []


def get_final_time(df):
    return df.index[-1].total_seconds() if df is not None else 0


def assign_start_values(df, trajectory_to_init_mapping, default_init=0):
    start_variables = {}
    for start_value_name,trajectory_name in trajectory_to_init_mapping.items():
        start_variables[start_value_name] = df[trajectory_name].iloc[-1] if trajectory_name and df is not None else default_init
    return start_variables

