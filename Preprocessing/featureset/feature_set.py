import pandas as pd
from typing import List
import numpy as np

from .feature import Feature


class FeatureSet:
    fmu_type = None
    features: List[Feature] = None

    def __init__(self, filename=None):
        if filename is not None:
            self.read_interface_file(filename)

    def read_interface_file(self, filename):
        """
        Read interface file - creates features
        @param filename: interface file
        """
        try:
            first_row = pd.read_csv(filename, nrows=1, sep=';')
            self.fmu_type = first_row.columns[0]
            data = pd.read_csv(filename, sep=';', encoding='latin-1', header=0, low_memory=False, skiprows=1)
            self.features = self._get_selected_features_from_file(data)
        except BaseException as ex:
            print(str(ex))

    ############################################# Static feature selection #############################################

    @staticmethod
    def get_selected_feats(features: List[Feature], value=None, selector="models"):
        return [feature for feature in features if value in getattr(feature, selector, []) or value is None] if features else []

    @staticmethod
    def get_feats_with_attr(features: List[Feature], attr="static"):
        return [feature for feature in features if getattr(feature, attr, False)] if features else []

    @staticmethod
    def get_feat_names(features):
        return [feature.name for feature in features]

    ######################################### Getters ##################################################################

    def get_dynamic_feats(self, model_name=None):
        return self.get_selected_feats_w_attr(model_name, selector="models", attr="dynamic")

    def get_static_feats(self, model_name=None):
        return self.get_selected_feats_w_attr(model_name, selector="models", attr="static")

    def get_output_feats(self, model_name=None):
        return self.get_selected_feats_w_attr(model_name, selector="models", attr="output")

    def get_input_feats(self, model_name=None):
        return self.get_selected_feats_w_attr(model_name, selector="models", attr="input")

    def get_param_feats(self, model_name=None):
        return self.get_selected_feats_w_attr(model_name, selector="models", attr="parameter")

    def get_output_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_output_feats(model_name))

    def get_static_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_static_feats(model_name))

    def get_dynamic_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_dynamic_feats(model_name))

    def get_input_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_input_feats(model_name))

    def get_dynamic_input_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_selected_feats_w_attrs(model_name, attrs=['input', 'dynamic']))

    def get_dynamic_output_feature_names(self, model_name=None):
        return self.get_feat_names(self.get_selected_feats_w_attrs(model_name, attrs=['output', 'dynamic']))

    def get_static_input_feature_names(self, model_name=""):
        return self.get_feat_names(self.get_selected_feats_w_attrs(model_name, attrs=['input', 'static']))

    def get_feature_by_name(self, name):
        for feature in self.features:
            if feature.name == name:
                return feature

    ########################### Add / remove features ##################################################################

    def add_feature(self, feature):
        self.features.append(feature)

    def add_cyclic_input_feature(self, name:str=""):
        self.add_feature(Feature(name=name, static=True, cyclic=True, input=True, init=0, models=self.get_output_feature_names()))

    def add_cyclic_input_features(self, names: List[str] = []):
        for name in names:
            self.add_cyclic_input_feature(name)

    def add_statistial_input_feature(self, name:str=""):
        self.add_feature(Feature(name=name, static=True, statistical=True, input=True, init=0, models=self.get_output_feature_names()))

    def add_statistical_input_features(self, names: List[str] = []):
        for name in names:
            self.add_statistial_input_feature(name)

    def remove_feature_by_name(self, name):
        for feature in self.features:
            if feature.name == name:
                self.features.remove(feature)
                break

    ############################## Helper methods #####################################################################

    def get_selected_feats_w_attr(self, value=None, selector="models", attr=None):
        return [feature for feature in self.features if
                (value in getattr(feature, selector, []) or value is None) and
                (getattr(feature, attr, False) or attr is None)]

    def get_selected_feats_w_attrs(self, value=None, selector="models", attrs: List[str]=None):
        list_feats = []
        for feat in self.features:
            if value in getattr(feat, selector, []) or value is None:
                if np.all(np.array([getattr(feat, attr, False) for attr in attrs])):
                    list_feats.append(feat)
        return list_feats

    @staticmethod
    def _get_selected_features_from_file(data, selector="", select_value=""):
        selected_data = data[data[selector] == select_value] if selector != "" else data
        selected_data = selected_data.fillna("")
        selected_data['Init'] = selected_data['Init'].astype('float')
        return [Feature(name=row["Name"],
                        models=row["Predictions"].split(','),
                        input=row["In_Out"] == 'input',
                        output=row["In_Out"] == 'output',
                        parameter=row["In_Out"] == 'parameter',
                        datatype=row["Type"],
                        static=row["Stat_Dyn"] == 'static',
                        dynamic=row["Stat_Dyn"] == "dynamic",
                        init=row["Init"],
                        description=row["Description"])
                for _,row in selected_data.iterrows()]