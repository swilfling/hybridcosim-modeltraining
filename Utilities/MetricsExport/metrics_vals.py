from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class MetricsVal:
    model_type: str = ""
    model_name: str = ""
    target_feat: str = ""
    expansion_type: str = ""
    featsel_thresh: str = ""
    metrics_type: str = ""
    metrics_name: str = ""
    usecase_name: str = ""
    val: float = 0.0

    def set_metr_properties(self, model_type="", model_name="", expansion_type="", featsel_thresh="", usecase_name=""):
        self.model_type = model_type
        self.model_name = model_name
        self.expansion_type = expansion_type
        self.featsel_thresh = featsel_thresh
        self.usecase_name = usecase_name

    def get_identifier(self, excluded_attribute=""):
        return "_".join(str(v) for k, v in self.__dict__.items() if k != excluded_attribute and k != "val")


class MetrValsSet:
    metr_vals: List[MetricsVal] = []

    ############################ Get, add, remove metrics vals #########################################################

    def add_metr_val(self, metr_val: MetricsVal):
        self.metr_vals.append(metr_val)

    def remove_metr_val(self, metr_val: MetricsVal):
        self.metr_vals.remove(metr_val)

    def add_metr_vals(self, metr_vals: List[MetricsVal]):
        self.metr_vals += metr_vals

    def get_metrs_of_attr(self, attr="metrics_type", val=""):
        return [metr_val for metr_val in self.metr_vals if getattr(metr_val, attr, "") == val or val == ""]

    @staticmethod
    def get_metrs_of_attr_from_list(metr_vals, attr="metrics_type", val=""):
        return [metr_val for metr_val in metr_vals if getattr(metr_val, attr, "") == val or val == ""]

    ######################################## Create dataframe ##########################################################

    def create_df_metrics(self, metrics_type="", index_col='model_type'):
        """
        Create dataframe from metrics.
        @param metrics_type: Select metrics of this type. If this param is empty, all metrics are selected.
        @return: dataframe containing metrics
        """
        metr_vals = self.get_metrs_of_attr("metrics_type", metrics_type)
        model_types = set(getattr(metr_val, index_col) for metr_val in metr_vals)
        df_metrs = pd.DataFrame()
        for model_type in model_types:
            metr_vals_model = MetrValsSet.get_metrs_of_attr_from_list(metr_vals, index_col, model_type)
            metr_data = {v.get_identifier(excluded_attribute=index_col):[v.val] for v in metr_vals_model}
            df_metr_cur = pd.DataFrame(data=metr_data, index=[model_type])
            df_metrs = df_metr_cur if df_metrs.empty else df_metrs.append(df_metr_cur)
        return df_metrs