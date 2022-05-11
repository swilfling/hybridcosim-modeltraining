import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class Feature:
    name: str = ""
    models: List[str] = None
    datatype: str = ""
    feature_type: str = ""
    static: bool = False
    dynamic: bool = False
    cyclic: bool = False
    statistical: bool = False
    init: float = None
    description: str = ""


class FeatureSet:
    fmu_type = None
    features: List[Feature] = None

    def __init__(self, filename=None):
        if filename is not None:
            self.read_interface_file(filename)

    def read_interface_file(self, filename):
        try:
            first_row = pd.read_csv(filename, nrows=1, sep=';')
            self.fmu_type = first_row.columns[0]
            data = pd.read_csv(filename, sep=';', encoding='latin-1', header=0, low_memory=False, skiprows=1)
            self.features = self._get_selected_features_from_file(data)
        except BaseException as ex:
            print(str(ex))
            return None

    @staticmethod
    def get_selected_feats(features: List[Feature], value=None, selector="models"):
        return [feature for feature in features if value in getattr(feature, selector, []) or value is None] if features else []

    @staticmethod
    def get_feats_with_attr(features: List[Feature], attr="static"):
        return [feature for feature in features if getattr(feature, attr, False)] if features else []

    @staticmethod
    def get_feat_names(features):
        return [feature.name for feature in features]

    def get_dynamic_feats(self, model_name=None):
        feats = self.get_feats_with_attr(self.features, "dynamic")
        return self.get_selected_feats(feats, model_name)

    def get_static_feats(self, model_name=None):
        feats = self.get_feats_with_attr(self.features, "static")
        return self.get_selected_feats(feats, model_name)

    def get_output_feats(self, model_name=None):
        feats = self.get_selected_feats(self.features, "output", "feature_type")
        return self.get_selected_feats(feats, model_name)

    def get_input_feats(self, model_name=None):
        feats = self.get_selected_feats(self.features, "input", "feature_type")
        return self.get_selected_feats(feats, model_name)

    def get_param_feats(self, model_name=None):
        feats = self.get_selected_feats(self.features, "parameter", "feature_type")
        return self.get_selected_feats(feats, model_name)

    def get_output_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_output_feats(model_name))

    def get_static_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_static_feats(model_name))

    def get_dynamic_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_dynamic_feats(model_name))

    def get_input_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_input_feats(model_name))

    def get_dynamic_input_feature_names(self, model_name=None):
        feats = self.get_feats_with_attr(self.features, "dynamic")
        feats = self.get_selected_feats(feats, "input", "feature_type")
        return self.get_feat_names(self.get_selected_feats(feats, model_name))

    def get_dynamic_output_feature_names(self, model_name=""):
        feats = self.get_feats_with_attr(self.features, "dynamic")
        feats = self.get_selected_feats(feats, "output", "feature_type")
        return self.get_feat_names(self.get_selected_feats(feats, model_name))

    def get_static_input_feature_names(self, model_name=""):
        feats = self.get_feats_with_attr(self.features, "static")
        feats = self.get_selected_feats(feats, "input", "feature_type")
        return self.get_feat_names(self.get_selected_feats(feats, model_name))

    def add_feature(self, feature):
        self.features.append(feature)

    def get_feature_by_name(self, name):
        for feature in self.features:
            if feature.name == name:
                return feature

    def remove_feature_by_name(self, name):
        for feature in self.features:
            if feature.name == name:
                self.features.remove(feature)
                break

    @staticmethod
    def _get_selected_features_from_file(data, selector="", select_value=""):
        selected_data = data[data[selector] == select_value] if selector != "" else data
        selected_data = selected_data.fillna("")
        selected_data['Init'] = selected_data['Init'].astype('float')
        return [Feature(name=row["Name"], models=row["Predictions"].split(','),
                        feature_type=row["In_Out"], datatype=row["Type"], static=row["Stat_Dyn"] == 'static',
                        dynamic=row["Stat_Dyn"] == "dynamic", init=row["Init"], description=row["Description"]) for _,row in selected_data.iterrows()]