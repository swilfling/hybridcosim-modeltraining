import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class Feature:
    name: str = ""
    models: List[str] = None
    datatype: str = ""
    feature_type: str = ""
    init: None = None
    description: str = ""


class FeatureSet:
    fmu_type = None
    static_features: List[Feature] = None
    dynamic_features: List[Feature] = None
    input_features: List[Feature] = None
    output_features: List[Feature] = None
    parameter_features: List[Feature] = None

    def __init__(self, filename=None):
        if filename is not None:
            self.read_interface_file(filename)

    def read_interface_file(self, filename):
        try:
            first_row = pd.read_csv(filename, nrows=1, sep=';')
            self.fmu_type = first_row.columns[0]
            data = pd.read_csv(filename, sep=';', encoding='latin-1', header=0, low_memory=False, skiprows=1)
            self.input_features = self._get_selected_features_from_file(data, "In_Out", "input")
            self.output_features = self._get_selected_features_from_file(data, "In_Out", "output")
            self.parameter_features = self._get_selected_features_from_file(data, "In_Out", "parameter")
            self.static_features = self._get_selected_features_from_file(data, "Stat_Dyn", "static")
            self.dynamic_features = self._get_selected_features_from_file(data, "Stat_Dyn", "dynamic")
        except Exception as ex:
            print(str(ex))
            return None

    @staticmethod
    def get_feature_names_for_model(features: List[Feature], model_name="", selector="models"):
        return [feature.name for feature in features if model_name in getattr(feature, selector, []) or model_name == ""] if features else []

    def get_output_feature_names(self, model_name=""):
        return self.get_feature_names_for_model(self.output_features, model_name)

    def get_static_feature_names(self, model_name=""):
        return self.get_feature_names_for_model(self.static_features, model_name)

    def get_dynamic_feature_names(self, model_name=""):
        return self.get_feature_names_for_model(self.dynamic_features, model_name)

    def get_input_feature_names(self, model_name=""):
        return self.get_feature_names_for_model(self.input_features, model_name)

    def get_all_features_for_model(self, model_name=""):
         return {"Static Features": self.get_static_feature_names(model_name),
                 "Dynamic Features": self.get_dynamic_feature_names(model_name),
                 "Inputs": self.get_input_feature_names(model_name),
                 "Outputs": self.get_output_feature_names(model_name)}

    @staticmethod
    def _get_selected_features_from_file(data, selector, select_value):
        selected_data = data[data[selector] == select_value]
        return [Feature(name=row["Name"], models=row["Predictions"],
                        feature_type=row["In_Out"], datatype=row["Type"],
                        init=row["Init"], description=row["Description"]) for _,row in selected_data.iterrows()]